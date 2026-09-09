const status = document.querySelector("#status");
const camera = document.querySelector("#camera");
const dot = document.querySelector("#status-dot");

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
