(function() {
  var canvas = document.getElementById('bgCanvas');
  var ctx = canvas.getContext('2d');
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var colors = ['#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#636EFA'];

  function R() { return Math.random(); }

  function draw() {
    var w = window.innerWidth;
    var h = document.documentElement.scrollHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // ~1 trace per 15000 px² of screen area
    var count = Math.round((w * h) / 30000);

    for (var i = 0; i < count; i++) {
      var cx = R() * w;
      var cy = R() * h;
      var color = colors[Math.floor(R() * colors.length)];
      var radius = 30 + R() * 120;
      var startAngle = R() * Math.PI * 2;
      var dir = R() < 0.5 ? 1 : -1;
      var totalSteps = 80 + Math.floor(R() * 60);
      var lw = 0.8 + R() * 1.2;
      var dotR = 1.5 + R() * 1.5;

      // Pick a random "frozen moment" along the lifecycle
      var life = 0.1 + R() * 0.7; // avoid extremes (fully faded)
      var baseAlpha = 0.3 + R() * 0.7; // random overall opacity
      var envelope = baseAlpha * (1 - life);
      var headPos = life * (totalSteps - 1);
      var headIdx = Math.floor(headPos);
      var tailLen = Math.floor(totalSteps / 3);
      var tailStart = Math.max(0, headIdx - tailLen);

      // Draw tail with gradient alpha
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      for (var j = tailStart; j < headIdx; j++) {
        var frac = (j - tailStart) / tailLen;
        var a1 = startAngle + dir * (j / totalSteps) * Math.PI * 2;
        var a2 = startAngle + dir * ((j + 1) / totalSteps) * Math.PI * 2;
        ctx.globalAlpha = frac * frac * envelope;
        ctx.beginPath();
        ctx.moveTo(cx + Math.cos(a1) * radius, cy + Math.sin(a1) * radius);
        ctx.lineTo(cx + Math.cos(a2) * radius, cy + Math.sin(a2) * radius);
        ctx.stroke();
      }

      // Draw dot at head
      if (headIdx < totalSteps) {
        var aHead = startAngle + dir * (headIdx / totalSteps) * Math.PI * 2;
        ctx.beginPath();
        ctx.arc(cx + Math.cos(aHead) * radius, cy + Math.sin(aHead) * radius, dotR, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.globalAlpha = Math.min(1, 1.5 * envelope);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  draw();

  var timer;
  window.addEventListener('resize', function() {
    clearTimeout(timer);
    timer = setTimeout(draw, 200);
  });
})();
