const status = document.querySelector("#status");
const camera = document.querySelector("#camera");
const dot = document.querySelector("#status-dot");
const sourceCards = document.querySelectorAll(".source-card");
const videoFile = document.querySelector("#video-file");
const videoSource = document.querySelector("#video-source");
const videoState = document.querySelector("#video-state");
const uploadCard = document.querySelector("#upload-source-card");

const socketProtocol = location.protocol === "https:" ? "wss" : "ws";
const socket = new WebSocket(`${socketProtocol}://${location.host}/ws`);

socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.type !== "stats") return;

  status.textContent = message.status;
  camera.textContent = message.camera_name || message.error || "No active camera";
  dot.classList.toggle("offline", message.status !== "online");
  document.querySelector("#fps").textContent = message.fps || "-";
  document.querySelector("#latency").textContent = message.inference_ms || "-";
  document.querySelector("#total").textContent = message.objects?.total ?? "-";
  document.querySelector("#classes").textContent = JSON.stringify(
    message.objects?.classes || {},
    null,
    2,
  );
};

socket.onclose = () => {
  status.textContent = "WebSocket disconnected";
  dot.classList.add("offline");
};

function setActiveSource(source) {
  sourceCards.forEach((card) => card.classList.toggle("active", card.dataset.source === source));
}

async function selectCamera() {
  await fetch("/source/camera", { method: "POST" });
  setActiveSource("camera");
  videoSource.textContent = "Source: Pantau Semar";
  videoState.textContent = "Latest frame only";
}

async function uploadVideo(file) {
  const form = new FormData();
  form.append("video", file);
  status.textContent = "Uploading video";
  dot.classList.remove("offline");
  videoState.textContent = "Uploading...";
  const response = await fetch("/upload-video", { method: "POST", body: form });
  if (!response.ok) {
    throw new Error((await response.json()).detail || "Upload failed");
  }
  setActiveSource("upload");
  videoSource.textContent = `Source: ${file.name}`;
  videoState.textContent = "Processing preview";
}

sourceCards.forEach((card) => {
  card.addEventListener("click", () => {
    if (card.dataset.source === "camera") {
      selectCamera().catch((error) => { status.textContent = error.message; });
      return;
    }
    videoFile?.click();
  });
});

uploadCard?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    videoFile?.click();
  }
});

videoFile.addEventListener("change", () => {
  const [file] = videoFile.files;
  if (!file) return;
  uploadVideo(file).catch((error) => {
    status.textContent = error.message;
    videoState.textContent = "Upload failed";
  });
});
