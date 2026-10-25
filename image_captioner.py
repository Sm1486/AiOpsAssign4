import io
import time
import socket
import zipfile
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from prometheus_client import Counter, Gauge, Histogram, Summary, Info, start_http_server, disable_created_metrics, REGISTRY

disable_created_metrics()

METRICS_PORT = 18000

@st.cache_resource
def init_metrics():
    metrics = {}
    metrics["images_processed_total"] = Counter("captioner_images_processed_total", "Total number of images processed", ["mode", "client_ip"])
    metrics["requests_total"] = Counter("captioner_requests_total", "Total requests recieved by the Streamlit app", ["mode"])
    metrics["errors_total"] = Counter("captioner_errors_total", "Total errors encountered during captioning", ["error_type", "mode"])
    metrics["active_requests"] = Gauge("captioner_active_requests", "Number of captioning requests currently in progress")
    metrics["model_memory_mb"] = Gauge("captioner_model_memory_mb", "Approximate resident memory used by the BLIP model in MB")
    metrics["bulk_queue_size"] = Gauge("captioner_bulk_queue_size", "Number of images remaining in the current bulk-upload queue")
    metrics["inference_latency"] = Histogram("captioner_inference_latency_seconds", "Time taken in seconds to caption a single image", ["mode"], buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0))
    metrics["zip_processing_latency"] = Histogram("captioner_zip_processing_latency_seconds", "Total time in seconds to process an entire ZIP batch", buckets=(5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0))
    metrics["caption_length_summary"] = Summary("captioner_caption_length_chars", "Summary of generated caption lengths in characters")
    metrics["request_duration_summary"] = Summary("captioner_request_duration_seconds", "The total request duration per mode")
    app_info = Info("captioner_build", "Image Captioner application metadata")
    app_info.info({"version": "1.0.0", "model": "Salesforce/blip-image-captioning-base", "author": "Sahithi", "build_date": "October 2026"})
    
    start_http_server(METRICS_PORT)
    return metrics

metrics = init_metrics()
images_processed_total = metrics["images_processed_total"]
requests_total = metrics["requests_total"]
errors_total = metrics["errors_total"]
active_requests = metrics["active_requests"]
model_memory_mb = metrics["model_memory_mb"]
bulk_queue_size = metrics["bulk_queue_size"]
inference_latency = metrics["inference_latency"]
zip_processing_latency = metrics["zip_processing_latency"]
caption_length_summary = metrics["caption_length_summary"]
request_duration_summary = metrics["request_duration_summary"]

def get_client_ip():
    try:
        ctx = st.runtime.scriptrunner.get_script_run_ctx()
        if ctx and hasattr(ctx, "session_id"):
            return ctx.session_id[:8]
    except Exception:
        pass
    return socket.gethostname()

@st.cache_resource()
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.eval()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_memory_mb.set(param_bytes / (1024 ** 2))
    return processor, model

def generate_caption(image: Image.Image, processor, model, mode: str, client_ip: str):
    active_requests.inc()
    t_start = time.time()
    try:
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(output[0], skip_special_tokens=True)
        elapsed = time.time() - t_start
        inference_latency.labels(mode=mode).observe(elapsed)
        caption_length_summary.observe(len(caption))
        images_processed_total.labels(mode=mode, client_ip=client_ip).inc()
        return caption

    except Exception as exc:
        errors_total.labels(error_type=type(exc).__name__, mode=mode).inc()
        raise exc
    
    finally:
        active_requests.dec()

def main():
    st.set_page_config(page_title="AI Image Captioner", layout="centered")
    st.title("Image Captioner")
    processor, model = load_model()
    client_ip = get_client_ip()
    mode = st.radio("Select mode", options=["Single Image", "Bulk (ZIP file)"], horizontal=True)
    st.divider()

    if mode == "Single Image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"], help="Accepts JPG, PNG, WebP, BMP")
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)
            if st.button("Generate Caption", type="primary"):
                requests_total.labels(mode="single").inc()
                with st.spinner("Generating caption…"):
                    with request_duration_summary.time():
                        caption = generate_caption(image, processor, model, "single", client_ip)
                st.success(f"**Caption:** {caption}")

    else:
        uploaded_zip = st.file_uploader("Upload a ZIP file of images", type=["zip"], help="ZIP should contain JPG/PNG/WebP images at the root level")
        if uploaded_zip is not None:
            if st.button("Process All Images", type="primary"):
                requests_total.labels(mode="bulk").inc()
                zip_start = time.time()
                results = []
                errors = []
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                    image_names = [n for n in zf.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
                if not image_names:
                    st.warning("No valid image files found in the ZIP.")
                else:
                    bulk_queue_size.set(len(image_names))
                    progress = st.progress(0, text="Starting the captioning")

                    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zf:
                        for idx, name in enumerate(image_names):
                            try:
                                img_bytes = zf.read(name)
                                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                caption = generate_caption(image, processor, model, "bulk", client_ip)
                                results.append({"file": name, "caption": caption})
                            except Exception as exc:
                                errors.append({"file": name, "error": str(exc)})

                            remaining = len(image_names) - (idx + 1)
                            bulk_queue_size.set(remaining)
                            pct = (idx + 1) / len(image_names)
                            progress.progress(pct, text=f"Processing {idx+1}/{len(image_names)}: {name}")

                    zip_elapsed = time.time() - zip_start
                    zip_processing_latency.observe(zip_elapsed)
                    bulk_queue_size.set(0)
                    progress.empty()
                    st.success(f"Processed **{len(results)}** images in **{zip_elapsed:.1f}s**" + (f" · {len(errors)} errors" if errors else ""))

                    if results:
                        st.subheader("Captions")
                        for r in results:
                            with st.expander(r["file"]):
                                st.write(r["caption"])

                    if errors:
                        st.subheader("Errors")
                        for e in errors:
                            st.error(f"**{e['file']}**: {e['error']}")

if __name__ == "__main__":
    main()
