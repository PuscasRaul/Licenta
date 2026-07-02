(() => {
    const SEND_W = 640;
    const SEND_INTERVAL = 700;
    const JPEG_QUALITY = 0.6;

    const video   = document.getElementById('video');
    const plateEl = document.getElementById('plate');
    const statusEl= document.getElementById('status');
    const fpsEl   = document.getElementById('fps');
    const btnStart= document.getElementById('start');
    const btnStop = document.getElementById('stop');
    const btnFlip = document.getElementById('flip');

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    let ws = null, stream = null, timer = null;
    let facing = 'environment';        // rear camera by default (plates)
    let recvTimes = [];

    const wsUrl = () =>
          (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws';

    function setStatus(t) { statusEl.innerHTML = t; }

    function showPlate(p) {
        if (p && p.trim()) { plateEl.textContent = p; plateEl.classList.remove('empty'); }
    }

    async function openCamera() {
        if (stream) stream.getTracks().forEach(t => t.stop());
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: facing }, width: { ideal: 1280 } },
            audio: false,
        });
        video.srcObject = stream;
        await video.play().catch(() => {});
    }

    function sendFrame() {
        if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
        const scale = Math.min(1, SEND_W / video.videoWidth);
        canvas.width  = Math.round(video.videoWidth  * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(b => {
            if (b && ws && ws.readyState === WebSocket.OPEN)
                b.arrayBuffer().then(a => ws.send(a));
        }, 'image/jpeg', JPEG_QUALITY);
    }

    function tickFps() {
        const now = performance.now();
        recvTimes.push(now);
        recvTimes = recvTimes.filter(t => now - t < 3000);
        const f = recvTimes.length > 1
              ? (recvTimes.length - 1) / ((recvTimes[recvTimes.length-1] - recvTimes[0]) / 1000)
              : 0;
        fpsEl.textContent = f.toFixed(1) + ' fps';
    }

    async function start() {
        btnStart.disabled = true;
        try {
            await openCamera();
        } catch (e) {
            setStatus('camera indisponibilă: ' + e.message);
            btnStart.disabled = false;
            return;
        }
        ws = new WebSocket(wsUrl());
        ws.binaryType = 'arraybuffer';
        setStatus('conectare…');

        ws.onopen = () => {
            setStatus('<b>în execuție</b>');
            btnStop.disabled = false;
            timer = setInterval(sendFrame, SEND_INTERVAL);
        };
        ws.onmessage = ev => {
            tickFps();
            let data; try { data = JSON.parse(ev.data); } catch { return; }
            showPlate(data.plate);
        };
        ws.onclose = () => stop();
        ws.onerror = () => setStatus('eroare de conexiune');
    }

    function stop() {
        if (timer) { clearInterval(timer); timer = null; }
        if (ws) { ws.onclose = null; try { ws.close(); } catch {} ws = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        video.srcObject = null;
        btnStart.disabled = false;
        btnStop.disabled = true;
        setStatus('oprit');
    }

    async function flip() {
        facing = facing === 'environment' ? 'user' : 'environment';
        if (stream) { try { await openCamera(); } catch (e) { setStatus(e.message); } }
    }

    btnStart.addEventListener('click', start);
    btnStop.addEventListener('click', stop);
    btnFlip.addEventListener('click', flip);
})();
