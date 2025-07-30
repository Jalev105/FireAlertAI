document.getElementById("detectBtn").addEventListener("click", () => {
  const input = document.getElementById("imageInput");
  const confidence = document.getElementById("confidence").value;
  const selectedModel = document.getElementById("modelSelect").value;

  if (input.files.length === 0) {
    alert("Please upload an image.");
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);
  formData.append("confidence", confidence);
  formData.append("model_name", selectedModel);

  fetch("http://localhost:5000/detect", {
    method: "POST",
    body: formData,
  })
    .then((res) => {
      if (!res.ok) throw new Error("Detection failed");
      return res.blob();
    })
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      document.getElementById("outputPreview").src = url;
      document.getElementById("saveBtn").href = url;
    })
    .catch((err) => {
      console.error(err);
      alert("Detection failed.");
    });
});

document.getElementById("confidence").addEventListener("input", (e) => {
  document.getElementById("confValue").textContent = e.target.value;
});

document.getElementById("imageInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const url = URL.createObjectURL(file);
    document.getElementById("inputPreview").src = url;
  }
});

fetch("http://localhost:5000/models")
  .then(res => res.json())
  .then(models => {
    const select = document.getElementById("modelSelect");
    models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      select.appendChild(opt);
    });
  });
