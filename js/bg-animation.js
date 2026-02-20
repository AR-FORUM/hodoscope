(function() {
  var canvas = document.getElementById('bgCanvas');
  var ctx = canvas.getContext('2d');
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var w, h;

  function resize() {
    w = window.innerWidth;
    h = document.documentElement.scrollHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  var colors = ['#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#636EFA'];
  var traces = [];

  function R() { return Math.random(); }

  function makeTrace() {
    var cx = R() * w;
    var cy = R() * h;
    var color = colors[Math.floor(R() * colors.length)];
    var radius = 30 + R() * 120;
    var startAngle = R() * Math.PI * 2;
    var dir = R() < 0.5 ? 1 : -1;
    var totalSteps = 120 + Math.floor(R() * 80);
    var lw = 0.8 + R() * 1.2;
    var opacity = 1.0;
    var dotR = 1.5 + R() * 1.5;

    var pts = [];
    for (var i = 0; i <= totalSteps; i++) {
      var a = startAngle + dir * (i / totalSteps) * Math.PI * 2;
      pts.push({x: cx + Math.cos(a) * radius, y: cy + Math.sin(a) * radius});
    }

    return {
      pts: pts, color: color, opacity: opacity, lw: lw, dotR: dotR,
      head: Math.floor(R() * totalSteps),
      tailLen: Math.floor(totalSteps / 3),
      speed: 0.08 + R() * 0.12,
      life: 0
    };
  }

  function initTraces() {
    traces = [];
    for (var i = 0; i < 80; i++) {
      var t = makeTrace();
      t.head = Math.floor(R() * t.pts.length);
      traces.push(t);
    }
  }

  var raf;
  function animate() {
    ctx.clearRect(0, 0, w, h);
    // Batch segments by color and quantized alpha to reduce state changes
    var batches = {};
    var dots = [];
    for (var i = traces.length - 1; i >= 0; i--) {
      var t = traces[i];
      t.head += t.speed;
      t.life = Math.min(t.head / (t.pts.length - 1), 1);
      if (t.life >= 1) { traces[i] = makeTrace(); traces[i].head = 0; continue; }
      var envelope = 1 - t.life;
      var headIdx = Math.floor(t.head);
      var tailStart = Math.max(0, headIdx - t.tailLen);
      for (var j = tailStart; j < headIdx; j++) {
        var frac = (j - tailStart) / t.tailLen;
        var alpha = t.opacity * frac * frac * envelope;
        var qa = (alpha * 10 + 0.5 | 0) / 10;
        if (qa <= 0) continue;
        var key = t.color + '|' + qa + '|' + t.lw;
        var b = batches[key];
        if (!b) { b = batches[key] = {color: t.color, alpha: qa, lw: t.lw, segs: []}; }
        b.segs.push(t.pts[j].x, t.pts[j].y, t.pts[j + 1].x, t.pts[j + 1].y);
      }
      if (headIdx < t.pts.length) {
        dots.push(t.pts[headIdx].x, t.pts[headIdx].y, t.dotR, t.color, t.opacity * 1.5 * envelope);
      }
    }
    // Draw batched line segments
    for (var key in batches) {
      var b = batches[key];
      ctx.strokeStyle = b.color;
      ctx.globalAlpha = b.alpha;
      ctx.lineWidth = b.lw;
      ctx.beginPath();
      var s = b.segs;
      for (var k = 0; k < s.length; k += 4) {
        ctx.moveTo(s[k], s[k + 1]);
        ctx.lineTo(s[k + 2], s[k + 3]);
      }
      ctx.stroke();
    }
    // Draw dots
    for (var k = 0; k < dots.length; k += 5) {
      ctx.beginPath();
      ctx.arc(dots[k], dots[k + 1], dots[k + 2], 0, Math.PI * 2);
      ctx.fillStyle = dots[k + 3];
      ctx.globalAlpha = dots[k + 4];
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(animate);
  }

  resize(); initTraces(); animate();

  var timer;
  window.addEventListener('resize', function() {
    clearTimeout(timer);
    timer = setTimeout(function() { resize(); initTraces(); }, 200);
  });
  document.addEventListener('visibilitychange', function() {
    if (document.hidden) { cancelAnimationFrame(raf); raf = null; }
    else if (!raf) { animate(); }
  });
})();
