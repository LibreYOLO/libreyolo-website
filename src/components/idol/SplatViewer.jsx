'use client'

import { useEffect, useRef, useState } from 'react'

/*
 * A minimal Gaussian-splat viewer for the Speridlabs Idol gallery. Each
 * Gaussian is drawn as a screen-space ellipse (its covariance projected
 * through the perspective Jacobian), sorted far to near and alpha-blended.
 * LibreYOLO's own code, ported from the viewer used for the Idol launch
 * video. It reads the .lyg files written by scripts/idol-gallery/build.py.
 */

const UP = [0, -1, 0] // object up in 3DGS files, which are y-down
const PER_ROW = 1024 // splats per data-texture row, 3 texels each

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

uniform usampler2D u_data;
uniform vec3 u_eye, u_right, u_down, u_fwd;
uniform vec2 u_focal, u_viewport, u_limit;

in vec2 a_corner;
in uint a_index;

out vec4 v_color;
out vec2 v_uv;

void main() {
  ivec2 base = ivec2(int(a_index & 1023u) * 3, int(a_index >> 10u));
  uvec4 t0 = texelFetch(u_data, base, 0);
  uvec4 t1 = texelFetch(u_data, base + ivec2(1, 0), 0);
  uvec4 t2 = texelFetch(u_data, base + ivec2(2, 0), 0);

  vec3 rel = uintBitsToFloat(t0.xyz) - u_eye;
  vec3 cam = vec3(dot(u_right, rel), dot(u_down, rel), dot(u_fwd, rel));
  if (cam.z < 0.01) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 center = u_focal * cam.xy / cam.z;
  if (any(greaterThan(abs(center), 0.65 * u_viewport + 64.0))) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }

  vec2 t = clamp(cam.xy / cam.z, -u_limit, u_limit);
  vec3 j0 = (u_focal.x / cam.z) * (u_right - t.x * u_fwd);
  vec3 j1 = (u_focal.y / cam.z) * (u_down - t.y * u_fwd);
  vec4 s = uintBitsToFloat(t1);
  vec2 s2 = uintBitsToFloat(t2.xy);
  mat3 cov = mat3(s.x, s.y, s.z, s.y, s.w, s2.x, s.z, s2.x, s2.y);
  float a = dot(j0, cov * j0) + 0.3;
  float b = dot(j0, cov * j1);
  float c = dot(j1, cov * j1) + 0.3;

  float mid = 0.5 * (a + c);
  float radius = length(vec2(0.5 * (a - c), b));
  float l1 = mid + radius;
  float l2 = max(mid - radius, 0.1);
  vec2 ea = vec2(b, l1 - a);
  vec2 eb = vec2(l1 - c, b);
  vec2 e1 = dot(ea, ea) > dot(eb, eb) ? ea : eb;
  e1 = dot(e1, e1) > 0.0 ? normalize(e1) : vec2(1.0, 0.0);
  vec2 e2 = vec2(-e1.y, e1.x);
  float r1 = min(3.0 * sqrt(l1), 1024.0);
  float r2 = min(3.0 * sqrt(l2), 1024.0);
  vec2 offset = a_corner.x * r1 * e1 + a_corner.y * r2 * e2;
  v_uv = a_corner * vec2(r1 / sqrt(l1), r2 / sqrt(l2));

  uint rgba = t0.w;
  v_color = vec4(float(rgba & 255u), float((rgba >> 8) & 255u), float((rgba >> 16) & 255u), float(rgba >> 24)) / 255.0;
  vec2 p = center + offset;
  gl_Position = vec4(2.0 * p.x / u_viewport.x, -2.0 * p.y / u_viewport.y, 0.0, 1.0);
}`

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 v_color;
in vec2 v_uv;
out vec4 frag;

void main() {
  float power = -0.5 * dot(v_uv, v_uv);
  if (power < -4.5) discard;
  float alpha = min(0.99, v_color.a * exp(power));
  if (alpha < 1.0 / 255.0) discard;
  frag = vec4(v_color.rgb * alpha, alpha);
}`

const clamp = (value, low, high) => Math.min(high, Math.max(low, value))
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
const normalize = (v) => { const n = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((x) => x / n) }

async function fetchObject(url, signal) {
  const response = await fetch(url, { signal })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  let bytes = new Uint8Array(await response.arrayBuffer())
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
    bytes = new Uint8Array(await new Response(stream).arrayBuffer())
  }
  return bytes.buffer
}

// Decode a .lyg file into the data texture: centre, RGBA8, and the six unique
// covariance entries R S S^T R^T, three RGBA32UI texels per Gaussian.
function decode(buffer) {
  const head = new DataView(buffer, 0, 40)
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4))
  if (magic !== 'LYG1') throw new Error('Not a LibreYOLO splat file.')
  const count = head.getUint32(4, true)
  const lo = [0, 1, 2].map((k) => head.getFloat32(8 + 4 * k, true))
  const hi = [0, 1, 2].map((k) => head.getFloat32(20 + 4 * k, true))
  const sMin = head.getFloat32(32, true)
  const sMax = head.getFloat32(36, true)
  let at = 40
  const pos = new Uint16Array(buffer, at, 3 * count); at += 6 * count
  const scale = new Uint8Array(buffer, at, 3 * count); at += 3 * count
  const rot = new Uint8Array(buffer, at, 4 * count); at += 4 * count
  const rgba = new Uint8Array(buffer, at, 4 * count)

  const rows = Math.max(1, Math.ceil(count / PER_ROW))
  const words = new Uint32Array(rows * PER_ROW * 12)
  const floats = new Float32Array(words.buffer)
  const positions = new Float32Array(3 * count)
  const step = (sMax - sMin) / 255
  const sample = []
  const every = Math.max(1, Math.floor(count / 30000))
  for (let i = 0; i < count; i++) {
    let w = rot[4 * i] - 128, x = rot[4 * i + 1] - 128, y = rot[4 * i + 2] - 128, z = rot[4 * i + 3] - 128
    const norm = Math.sqrt(w * w + x * x + y * y + z * z) || 1
    w /= norm; x /= norm; y /= norm; z /= norm
    const sx = Math.exp(sMin + step * scale[3 * i])
    const sy = Math.exp(sMin + step * scale[3 * i + 1])
    const sz = Math.exp(sMin + step * scale[3 * i + 2])
    const m00 = (1 - 2 * (y * y + z * z)) * sx, m01 = 2 * (x * y - w * z) * sy, m02 = 2 * (x * z + w * y) * sz
    const m10 = 2 * (x * y + w * z) * sx, m11 = (1 - 2 * (x * x + z * z)) * sy, m12 = 2 * (y * z - w * x) * sz
    const m20 = 2 * (x * z - w * y) * sx, m21 = 2 * (y * z + w * x) * sy, m22 = (1 - 2 * (x * x + y * y)) * sz
    const o = 12 * i
    for (let k = 0; k < 3; k++) {
      const v = lo[k] + (pos[3 * i + k] / 65535) * (hi[k] - lo[k])
      floats[o + k] = v
      positions[3 * i + k] = v
    }
    words[o + 3] = rgba[4 * i] | (rgba[4 * i + 1] << 8) | (rgba[4 * i + 2] << 16) | (rgba[4 * i + 3] << 24)
    floats[o + 4] = m00 * m00 + m01 * m01 + m02 * m02
    floats[o + 5] = m00 * m10 + m01 * m11 + m02 * m12
    floats[o + 6] = m00 * m20 + m01 * m21 + m02 * m22
    floats[o + 7] = m10 * m10 + m11 * m11 + m12 * m12
    floats[o + 8] = m10 * m20 + m11 * m21 + m12 * m22
    floats[o + 9] = m20 * m20 + m21 * m21 + m22 * m22
    if (rgba[4 * i + 3] > 51 && i % every === 0) sample.push(i)
  }

  // Orbit around the centre of a robust bounding box of the opaque splats.
  let center = [0, 0, 0]
  let radius = 1
  if (sample.length) {
    const low = [], high = []
    for (let k = 0; k < 3; k++) {
      const values = Float32Array.from(sample, (s) => positions[3 * s + k]).sort()
      low.push(values[Math.floor(0.005 * (values.length - 1))])
      high.push(values[Math.ceil(0.995 * (values.length - 1))])
    }
    center = low.map((v, k) => (v + high[k]) / 2)
    radius = 0.5 * Math.hypot(high[0] - low[0], high[1] - low[1], high[2] - low[2]) || 1
  }
  return { count, rows, words, positions, center, radius }
}

function createRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: true })
  if (!gl) return null
  const compile = (type, source) => {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader))
    return shader
  }
  const program = gl.createProgram()
  gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX_SHADER))
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT_SHADER))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program))
  const uniform = {}
  for (const name of ['u_data', 'u_eye', 'u_right', 'u_down', 'u_fwd', 'u_focal', 'u_viewport', 'u_limit']) {
    uniform[name] = gl.getUniformLocation(program, name)
  }
  const vao = gl.createVertexArray()
  gl.bindVertexArray(vao)
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer())
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW)
  const corner = gl.getAttribLocation(program, 'a_corner')
  gl.enableVertexAttribArray(corner)
  gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0)
  const orderBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, orderBuffer)
  const index = gl.getAttribLocation(program, 'a_index')
  gl.enableVertexAttribArray(index)
  gl.vertexAttribIPointer(index, 1, gl.UNSIGNED_INT, 0, 0)
  gl.vertexAttribDivisor(index, 1)
  const texture = gl.createTexture()
  return { gl, program, uniform, vao, orderBuffer, texture }
}

export default function SplatViewer({ src, camera, active = true, label }) {
  const canvasRef = useRef(null)
  const stateRef = useRef(null)
  const [status, setStatus] = useState('idle')

  // Renderer and render loop: created once per mounted canvas.
  useEffect(() => {
    const canvas = canvasRef.current
    let renderer
    try {
      renderer = createRenderer(canvas)
    } catch {
      renderer = null
    }
    if (!renderer) {
      setStatus('nowebgl')
      return undefined
    }
    const { gl } = renderer
    const s = {
      renderer, scene: null, dirty: true, spinning: true, spinFrom: 0, last: 0, lastSortFwd: null,
      order: null, depths: null, keys: null, buckets: new Uint32Array(65536), drag: null, visible: true,
      view: { yaw: 0, pitch: 0.2, distance: 2, fov: 0.7, target: [0, 0, 0] },
    }
    stateRef.current = s

    const basis = () => {
      const { view } = s
      const cp = Math.cos(view.pitch)
      const dir = [Math.sin(view.yaw) * cp, -Math.sin(view.pitch), Math.cos(view.yaw) * cp]
      const eye = view.target.map((v, k) => v + view.distance * dir[k])
      const fwd = dir.map((v) => -v)
      const right = normalize(cross(fwd, UP))
      const down = cross(fwd, right)
      return { eye, fwd, right, down }
    }
    s.basis = basis

    const sort = (fwd) => {
      const n = s.scene.count, p = s.scene.positions
      let min = Infinity, max = -Infinity
      for (let i = 0; i < n; i++) {
        const d = fwd[0] * p[3 * i] + fwd[1] * p[3 * i + 1] + fwd[2] * p[3 * i + 2]
        s.depths[i] = d
        if (d < min) min = d
        if (d > max) max = d
      }
      const scale = 65535 / Math.max(max - min, 1e-9)
      s.buckets.fill(0)
      for (let i = 0; i < n; i++) {
        const key = ((max - s.depths[i]) * scale) | 0
        s.keys[i] = key
        s.buckets[key]++
      }
      for (let k = 0, sum = 0; k < 65536; k++) {
        const size = s.buckets[k]
        s.buckets[k] = sum
        sum += size
      }
      for (let i = 0; i < n; i++) s.order[s.buckets[s.keys[i]]++] = i
      gl.bindBuffer(gl.ARRAY_BUFFER, renderer.orderBuffer)
      gl.bufferData(gl.ARRAY_BUFFER, s.order, gl.DYNAMIC_DRAW)
    }

    let frame = 0
    const draw = (now) => {
      frame = requestAnimationFrame(draw)
      if (!s.visible) return
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const width = Math.round(canvas.clientWidth * dpr)
      const height = Math.round(canvas.clientHeight * dpr)
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width
        canvas.height = height
        s.dirty = true
      }
      const dt = s.last ? Math.min(0.1, (now - s.last) / 1000) : 0
      s.last = now
      if (s.scene && s.spinning && !s.drag && now > s.spinFrom) {
        s.view.yaw += 0.3 * dt
        s.dirty = true
      }
      if (!s.dirty) return
      s.dirty = false
      gl.viewport(0, 0, width, height)
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      if (!s.scene) return
      const { eye, fwd, right, down } = basis()
      if (!s.lastSortFwd || dot(fwd, s.lastSortFwd) < 0.99995) {
        sort(fwd)
        s.lastSortFwd = fwd
      }
      const focal = height / 2 / Math.tan(s.view.fov / 2)
      gl.useProgram(renderer.program)
      gl.bindVertexArray(renderer.vao)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, renderer.texture)
      const u = renderer.uniform
      gl.uniform1i(u.u_data, 0)
      gl.uniform3fv(u.u_eye, eye)
      gl.uniform3fv(u.u_right, right)
      gl.uniform3fv(u.u_down, down)
      gl.uniform3fv(u.u_fwd, fwd)
      gl.uniform2f(u.u_focal, focal, focal)
      gl.uniform2f(u.u_viewport, width, height)
      gl.uniform2f(u.u_limit, (1.3 * width) / 2 / focal, (1.3 * height) / 2 / focal)
      gl.disable(gl.DEPTH_TEST)
      gl.enable(gl.BLEND)
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.scene.count)
    }
    frame = requestAnimationFrame(draw)

    const observer = new IntersectionObserver(([entry]) => {
      s.visible = entry.isIntersecting
      s.dirty = true
    })
    observer.observe(canvas)

    const onDown = (event) => {
      s.drag = { x: event.clientX, y: event.clientY, pan: event.button !== 0 || event.shiftKey }
      canvas.setPointerCapture(event.pointerId)
      s.spinning = false
    }
    const onMove = (event) => {
      if (!s.drag || !s.scene) return
      const dx = event.clientX - s.drag.x
      const dy = event.clientY - s.drag.y
      s.drag.x = event.clientX
      s.drag.y = event.clientY
      if (s.drag.pan) {
        const { right, down } = basis()
        const perPixel = (2 * s.view.distance * Math.tan(s.view.fov / 2)) / canvas.clientHeight
        for (let k = 0; k < 3; k++) s.view.target[k] -= (right[k] * dx + down[k] * dy) * perPixel
      } else {
        s.view.yaw += dx * 0.006
        s.view.pitch = clamp(s.view.pitch + dy * 0.006, -1.45, 1.45)
      }
      s.dirty = true
    }
    const onUp = () => { s.drag = null }
    // Plain scrolling keeps scrolling the page; a pinch, or ctrl/cmd + scroll, zooms.
    const onWheel = (event) => {
      if (!s.scene || !(event.ctrlKey || event.metaKey)) return
      event.preventDefault()
      s.view.distance = clamp(s.view.distance * Math.exp(event.deltaY * 0.004), 0.3 * s.scene.radius, 30 * s.scene.radius)
      s.spinning = false
      s.dirty = true
    }
    const onDouble = () => s.reset && s.reset()
    const onContext = (event) => event.preventDefault()
    canvas.addEventListener('pointerdown', onDown)
    canvas.addEventListener('pointermove', onMove)
    canvas.addEventListener('pointerup', onUp)
    canvas.addEventListener('pointercancel', onUp)
    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('dblclick', onDouble)
    canvas.addEventListener('contextmenu', onContext)

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      canvas.removeEventListener('pointerdown', onDown)
      canvas.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerup', onUp)
      canvas.removeEventListener('pointercancel', onUp)
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('dblclick', onDouble)
      canvas.removeEventListener('contextmenu', onContext)
      stateRef.current = null
      // Free the GPU memory only once the canvas has left the page: React
      // can run this cleanup and the effect again on the same canvas, and a
      // lost context cannot be reused.
      setTimeout(() => {
        if (!canvas.isConnected) gl.getExtension('WEBGL_lose_context')?.loseContext()
      }, 0)
    }
  }, [])

  // Load the requested object once the viewer is wanted.
  useEffect(() => {
    const s = stateRef.current
    if (!s || !src || !active) return undefined
    const controller = new AbortController()
    setStatus('loading')
    fetchObject(src, controller.signal)
      .then((buffer) => {
        if (controller.signal.aborted || !stateRef.current) return
        const next = decode(buffer)
        const { gl, texture } = s.renderer
        gl.bindTexture(gl.TEXTURE_2D, texture)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, PER_ROW * 3, next.rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, next.words)
        next.words = null
        s.order = new Uint32Array(next.count)
        s.depths = new Float32Array(next.count)
        s.keys = new Uint16Array(next.count)
        s.scene = next
        s.lastSortFwd = null
        // Start from the view Idol inferred for the photo, then turn slowly.
        s.reset = () => {
          const cam = camera || {}
          const fov = Number.isFinite(cam.fov) ? clamp(cam.fov, 25, 60) : 40
          s.view.yaw = Number.isFinite(cam.yaw) ? cam.yaw : 0
          s.view.pitch = clamp(Number.isFinite(cam.pitch) ? cam.pitch : 0.2, -1.45, 1.45)
          s.view.fov = (fov * Math.PI) / 180
          s.view.target = s.scene.center.slice()
          const canvas = canvasRef.current
          const halfY = s.view.fov / 2
          const halfX = Math.atan(Math.tan(halfY) * (canvas.clientWidth / Math.max(1, canvas.clientHeight)))
          s.view.distance = (1.15 * s.scene.radius) / Math.sin(Math.min(halfX, halfY))
          s.dirty = true
        }
        s.reset()
        s.spinning = true
        s.spinFrom = performance.now() + 1500
        setStatus('ready')
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setStatus('error')
      })
    return () => controller.abort()
  }, [src, camera, active])

  return (
    <div className="relative h-full w-full">
      <canvas
        ref={canvasRef}
        className="block h-full w-full cursor-grab active:cursor-grabbing"
        style={{ touchAction: 'pan-y' }}
        aria-label={label}
      />
      {status !== 'ready' && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-surface-400">
          {status === 'loading' && 'Loading 3D object...'}
          {status === 'error' && 'The 3D object could not be loaded.'}
          {status === 'nowebgl' && 'This browser has no WebGL2, which the 3D viewer needs.'}
        </div>
      )}
    </div>
  )
}
