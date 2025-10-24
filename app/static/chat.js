const uploadBtn = document.getElementById('upload');
const fileInput = document.getElementById('file_input');
const askBtn = document.getElementById('ask');
const queryInput = document.getElementById('query');
const out = document.getElementById('out');

uploadBtn.onclick = async () => {
  if(!fileInput.files.length) { alert('select file'); return }
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);
  const res = await fetch('/upload', {method: 'POST', body: fd});
  const j = await res.json();
  out.innerText = JSON.stringify(j, null, 2);
}

askBtn.onclick = async () => {
  const q = queryInput.value;
  const res = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({query: q})});
  const j = await res.json();
  out.innerText = j.answer || JSON.stringify(j, null, 2);
}
