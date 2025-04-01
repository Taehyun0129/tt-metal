import os
import sys
import ast
import time
import random
from pathlib import Path
import asyncio
import torch
import ttnn
from ttnn import ConcatMeshToTensor
from loguru import logger
from PIL import Image
from transformers import AutoImageProcessor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.websockets import WebSocketState
from starlette.templating import Jinja2Templates
from models.utility_functions import profiler
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


logger.remove()
logger.add(sys.stdout, level="INFO", filter=lambda record: record["level"].name != "WARNING")
app = FastAPI()


# StaticFiles mount
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/val_images", StaticFiles(directory=str(BASE_DIR / "val_images")), name="val_images")

################################################################################
# Global variables
################################################################################
mesh_device = None
test_infra = None
big_tensors = None
big_paths = []
is_ready = False

connected_websockets = set()
GLOBAL_BATCH_SIZE = 16
NUM_WARMUP_ITER = 10
NUM_MEASURE_ITER = 15
warmup_done = False
NUM_BATCHES_PER_CYCLE = 100

pending_outputs = []
trace_context = None

# Cycle statistics (reset per cycle) & cycle id
cycle_total = 0
cycle_correct = 0
cycle_id = 0


################################################################################
# Utility functions
################################################################################
def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


def dump_device_profiler(device):
    if isinstance(device, ttnn.Device):
        ttnn.DumpDeviceProfiler(device)
    else:
        for dev in device.get_device_ids():
            ttnn.DumpDeviceProfiler(device.get_device(dev))


ttnn.dump_device_profiler = dump_device_profiler

model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}

# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def imagenet_label_dict():
    with open("/opt/tt-metal/models/sample_data/imagenet_class_labels.txt", "r") as f:
        return ast.literal_eval(f.read())


def get_expected_label_from_filename(filename):
    wnid = filename.split("_")[-1].split(".")[0]
    return IMAGENET2012_CLASSES.get(wnid, wnid)


def run_model(device, tt_inputs, test_infra, warmup_iterations, measure_iterations, image_paths):
    import torch.nn.functional as F

    global warmup_done

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device, tt_inputs.to_torch())

    if not warmup_done:
        test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
        _ = ttnn.from_device(test_infra.run(), blocking=True)
        for _ in range(warmup_iterations):
            test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
            _ = ttnn.from_device(test_infra.run(), blocking=True)
        warmup_done = True

    ttnn.synchronize_device(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    tt_out = test_infra.run()
    tt_out = ttnn.from_device(tt_out, blocking=True)
    ttnn.synchronize_device(device)

    torch_tensor = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0)).float()
    logits = torch_tensor[:, 0, 0, :]

    label_dict = imagenet_label_dict()
    prediction_probs = F.softmax(logits, dim=-1)
    prediction = prediction_probs.argmax(dim=-1)
    results = []
    for i, (p_idx, path) in enumerate(zip(prediction.tolist(), image_paths)):
        pred_label = label_dict[p_idx]
        exp_label = get_expected_label_from_filename(Path(path).name)
        correct = exp_label in pred_label
        results.append(
            {"pred_label": pred_label, "expected_label": exp_label, "correct": correct, "path": Path(path).name}
        )
    return results


def prepare_trace_2cq(device, tt_inputs, test_infra, image_paths, warmup_iterations):
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        device, tt_inputs.to_torch()
    )
    tt_image_res = tt_inputs_host.to(device, sharded_mem_config_DRAM)
    op_event = ttnn.record_event(device, 0)

    # Compile
    profiler.start("compile")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    spec = test_infra.input_tensor.spec
    op_event = ttnn.record_event(device, 0)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("compile")
    ttnn.dump_device_profiler(device)

    # Cache
    profiler.start("cache")
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.output_tensor.deallocate(force=True)
    _ = ttnn.from_device(test_infra.run(), blocking=True)
    profiler.end("cache")
    ttnn.dump_device_profiler(device)

    # Trace Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    test_infra.output_tensor.deallocate(force=True)
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    tt_output_res = test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)
    ttnn.dump_device_profiler(device)

    # Warmup
    for _ in range(warmup_iterations):
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.dump_device_profiler(device)
    ttnn.synchronize_device(device)
    return {
        "tt_inputs_host": tt_inputs_host,
        "tt_image_res": tt_image_res,
        "input_mem_config": input_mem_config,
        "input_tensor": input_tensor,
        "tt_output_res": tt_output_res,
        "tid": tid,
        "op_event": op_event,
    }


def run_trace_2cq_model_inference(
    device, tt_inputs, test_infra, warmup_iterations, measure_iterations, image_paths, inference_idx
):
    global trace_context, pending_outputs
    import torch.nn.functional as F
    from ttnn import ConcatMeshToTensor
    from pathlib import Path

    if trace_context is None:
        trace_context = prepare_trace_2cq(device, tt_inputs, test_infra, image_paths, warmup_iterations)

    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("============ Inference Start ============")
    total_start = time.perf_counter()
    test_start = time.perf_counter()
    tt_inputs_host, _, _ = test_infra.setup_dram_sharded_input(device, tt_inputs.to_torch())
    test_end = time.perf_counter()
    logger.info(f"[Setup DRAM Sharded Input]] {test_end - test_start:.6f} sec")
    tt_image_res = trace_context["tt_image_res"]
    input_mem_config = trace_context["input_mem_config"]
    input_tensor = trace_context["input_tensor"]
    tt_output_res = trace_context["tt_output_res"]
    tid = trace_context["tid"]
    op_event = trace_context["op_event"]

    # Inference
    transfer_start = time.perf_counter()
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    transfer_end = time.perf_counter()
    logger.info(f"[Host → DRAM copy] {transfer_end - transfer_start:.6f} sec")

    reshard_start = time.perf_counter()
    input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
    op_event = ttnn.record_event(device, 0)
    reshard_end = time.perf_counter()
    logger.info(f"[Reshard to input_tensor] {reshard_end - reshard_start:.6f} sec")

    exec_start = time.perf_counter()
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    exec_end = time.perf_counter()
    logger.info(f"[Execute Trace] {exec_end - exec_start:.6f} sec")

    fetch_start = time.perf_counter()
    tt_out = ttnn.from_device(tt_output_res, blocking=False)
    pending_outputs.append((tt_out, image_paths))
    fetch_end = time.perf_counter()
    logger.info(f"[Fetch from device] {fetch_end - fetch_start:.6f} sec")

    results = []

    logger.info(f"[inference_idx : ] {inference_idx} ")
    # if inference_idx % 10 == 9:
    if (inference_idx + 1) % 100 == 0:
        ttnn.synchronize_device(device)

        all_logits = []
        all_image_paths = []

        for tt_out, image_paths in pending_outputs:
            torch_tensor = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(device, dim=0)).float()

            logits = torch_tensor[:, 0, 0, :]
            all_logits.append(logits)
            all_image_paths.extend(image_paths)

        logits = torch.cat(all_logits, dim=0)
        prediction_probs = F.softmax(logits, dim=-1)
        prediction = prediction_probs.argmax(dim=-1)

        label_dict = imagenet_label_dict()
        for i, (p_idx, path) in enumerate(zip(prediction.tolist(), all_image_paths)):
            pred_label = label_dict[p_idx]
            exp_label = get_expected_label_from_filename(Path(path).name)
            correct = exp_label in pred_label
            results.append(
                {"pred_label": pred_label, "expected_label": exp_label, "correct": correct, "path": Path(path).name}
            )

        pending_outputs.clear()

    total_end = time.perf_counter()
    logger.info(f"== Total Inference Time: {total_end - total_start:.6f} sec ==")
    return results


################################################################################
# Startup: Device/Model initialization and pre-processing of up to 3200 images
################################################################################
@app.on_event("startup")
async def on_startup():
    global mesh_device, test_infra, big_tensors, big_paths, is_ready

    logger.info("[Startup] Initializing device and loading 3200 images...")

    from tests.scripts.common import get_updated_device_params

    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"
    device_params = {"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1332224}
    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    if fabric_config:
        ttnn.initialize_fabric_config(fabric_config)

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 2), **updated_device_params)
    for d in dev.get_devices():
        d.enable_program_cache()
        # d.enable_async(True)

    mesh_device = dev

    def model_location_generator(model_version, model_subdir=""):
        folder = Path("/opt/tt-metal-models") / "tt_dnn-models" / model_subdir
        model_path = folder / f"{model_version}.pt"
        if not model_path.exists():
            import torchvision.models as models

            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            os.makedirs(folder, exist_ok=True)
            torch.save(model.state_dict(), model_path)
        return str(model_path)

    test_infra = create_test_infra(
        dev,
        GLOBAL_BATCH_SIZE,
        model_config["ACTIVATIONS_DTYPE"],
        model_config["WEIGHTS_DTYPE"],
        model_config["MATH_FIDELITY"],
        use_pretrained_weight=True,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )

    # Pre-processing: load at most 3200 images
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    data_dir = "/opt/tt-metal/resnet_demo/val_images"
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".JPEG")]
    if len(all_files) == 0:
        logger.warning("No JPEG images found.")
        return

    if len(all_files) > 3200:
        all_files = random.sample(all_files, 3200)

    raw_tensors = []
    raw_paths = []
    for p in all_files:
        try:
            img = Image.open(p).convert("RGB")
            inp = processor(img, return_tensors="pt")["pixel_values"].bfloat16()
            raw_tensors.append(inp)
            raw_paths.append(p)
        except Exception:
            pass

    if not raw_tensors:
        logger.warning("No valid images after preprocessing.")
        return

    big_tensors = torch.cat(raw_tensors, dim=0)
    big_paths = raw_paths
    is_ready = True
    logger.info(f"[Startup] Completed. {len(big_paths)} images loaded.")


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "image_paths": [Path(p).name for p in big_paths]}
    )


@app.get("/ready")
def ready():
    return {"ready": is_ready}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket is Conected!")
    connected_websockets.add(websocket)
    cycle_num = 0
    try:
        while True:
            cycle_num += 1
            await run_one_cycle_3200(cycle_num, websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket is Disconected.")
    finally:
        connected_websockets.discard(websocket)


def run_batch_inference(batch_idx):
    global mesh_device, test_infra, big_tensors, big_paths

    nd = mesh_device.get_num_devices()
    batch_size = GLOBAL_BATCH_SIZE * nd

    N = len(big_paths)
    chosen_idx = random.sample(range(N), batch_size)
    batch_tensor = big_tensors[torch.tensor(chosen_idx)]
    batch_files = [big_paths[i] for i in chosen_idx]

    return run_trace_2cq_model_inference(
        mesh_device,
        ttnn.from_torch(batch_tensor, dtype=ttnn.bfloat16),
        test_infra,
        NUM_WARMUP_ITER if not warmup_done else 0,
        NUM_MEASURE_ITER,
        batch_files,
        inference_idx=batch_idx,
    )


async def run_one_cycle_3200(cycle_num, websocket):
    global cycle_total, cycle_correct, cycle_id
    cycle_total = 0
    cycle_correct = 0
    cycle_id = cycle_num

    start_time = time.time()
    all_results = []

    for batch_idx in range(NUM_BATCHES_PER_CYCLE):
        results = await asyncio.to_thread(run_batch_inference, batch_idx)
        all_results.extend(results)

        cycle_correct += sum(r["correct"] for r in results)
        cycle_total += len(results)

    elapsed_time = time.time() - start_time
    accuracy = round(100.0 * cycle_correct / cycle_total, 2)
    fps = round(cycle_total / elapsed_time, 2)

    message = {
        "partial": False,
        "final": True,
        "cycle": cycle_id,
        "stats": {"total": cycle_total, "accuracy": accuracy, "fps": fps, "elapsed_time": elapsed_time},
        "results": [
            {
                "pred_label": r["pred_label"],
                "expected_label": r["expected_label"],
                "correct": r["correct"],
                "path": r["path"],
            }
            for r in all_results[-32:]
        ],
    }

    try:
        await websocket.send_json(message)
    except Exception as e:
        logger.error(f"Websocket Could not send Message!! error!!!!: {e}")

    logger.info(
        f"Finished cycle {cycle_id} (3200 images). Accuracy: {accuracy}%, FPS: {fps}, Elapsed_time : {elapsed_time}"
    )


async def broadcast_json(msg):
    dead_ws = []
    for ws in connected_websockets:
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.send_json(msg)
            except Exception:
                dead_ws.append(ws)
    for ws in dead_ws:
        connected_websockets.discard(ws)
