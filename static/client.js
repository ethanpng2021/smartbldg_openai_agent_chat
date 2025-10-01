// static/client.js
const socket = io();
let conversationId = null;

socket.on("connect", () => {
  socket.emit("start");
});

socket.on("session", (data) => {
  conversationId = data.conversation_id;
});

socket.on("agent_reply", (data) => {
  const messages = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = "msg " + data.type;
  div.innerHTML = `<span class="${data.type}">${data.type === "agent" ? "Agent" : data.type}:</span> ${data.message}`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
});

window.sendMessage = () => {
  const input = document.getElementById("input");
  const message = input.value;
  if (!message.trim()) return;
  socket.emit("message", { conversation_id: conversationId, message });
  const messages = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = "msg user";
  div.innerHTML = `<span class="user">You:</span> ${message}`;
  messages.appendChild(div);
  input.value = "";
};
