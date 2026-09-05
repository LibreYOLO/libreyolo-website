"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play, Search, X } from "lucide-react";
import { Link } from "@/i18n/navigation";
import styles from "./explorer.module.css";

const descriptions = {
  detect:
    "Detect and classify objects in images and video. Returns bounding boxes, class labels and confidence scores.",
  segment: "Get a separate pixel mask for each detected object.",
  pose: "Detect joints and landmarks on each person. Returns keypoint coordinates and confidence scores.",
  classify:
    "Assign a label to an image. Choose a trained classifier or use a zero-shot model with your own class names.",
  obb: "Locate objects with rotated boxes, especially in aerial imagery where orientation matters.",
  depth:
    "Estimate relative depth from a single image. Returns a dense depth map.",
  semantic:
    "Assign a class to each pixel: road, building, vegetation, and other regions of a scene.",
  panoptic:
    "Combine scene-level classes with individual object masks in one prediction.",
  point:
    "Find object centres when you need locations or counts without full bounding boxes.",
  ocr: "Locate text regions and read their contents. Access polygons, recognized text and confidence scores.",
  embed:
    "Turn images or regions into vectors for similarity search, retrieval and recognition.",
  gaze: "Estimate where a person is looking from an image of their face.",
  edge: "Extract boundaries and fine structure as a dense edge map.",
  normal: "Estimate the orientation of visible surfaces, pixel by pixel.",
  matte:
    "Separate a foreground subject from its background with a soft alpha matte.",
  restore:
    "Restore or upscale images with specialist models. The supported operation depends on the model.",
  mesh: "Recover a 3D body mesh from a person in an image.",
};

function RecordedVideo({ src, poster, label, eager = false }) {
  const video = useRef(null);
  const userPaused = useRef(false);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    const element = video.current;
    if (
      !eager ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ||
      navigator.connection?.saveData
    )
      return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !userPaused.current)
          element.play().catch(() => {});
        else element.pause();
      },
      { threshold: 0.2 },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [src, eager]);
  return (
    <div className={styles.videoWrap}>
      <video
        ref={video}
        src={src}
        poster={poster}
        muted
        loop
        playsInline
        preload={eager ? "metadata" : "none"}
        aria-label={label}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
      />
      <button
        type="button"
        className={styles.play}
        onClick={() => {
          userPaused.current = playing;
          if (playing) video.current.pause();
          else video.current.play().catch(() => {});
        }}
        aria-label={playing ? `Pause ${label}` : `Play ${label}`}
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
        <span>{playing ? "Pause" : "Play"}</span>
      </button>
    </div>
  );
}


export default function TaskExplorer({ tasks }) {
  const [active, setActive] = useState("segment");
  const [query, setQuery] = useState("");
  const selected = tasks.find((t) => t.task === active) || tasks[0];
  const models = selected.models.filter((model) =>
    model.name.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <div className={styles.home}>
      <section className={`${styles.section} ${styles.explorer}`} id="tasks">
        <div className={styles.sectionIntro}>
          <h2>
            One API.
            <br />
            {tasks.length} tasks.
          </h2>
          <p>
            Detection, segmentation, pose, depth, OCR and more. Select a task to
            see the model output and available models.
          </p>
        </div>
        <div className={styles.taskButtons} aria-label="Choose a vision task">
          {tasks.map((task) => (
            <button
              type="button"
              key={task.task}
              aria-pressed={active === task.task}
              onClick={() => {
                setActive(task.task);
                setQuery("");
              }}
            >
              {task.label}
            </button>
          ))}
        </div>
        <div className={styles.taskDetail}>
          <div className={styles.taskVisual}>
            {selected.video ? (
              <RecordedVideo
                key={selected.task}
                src={selected.video}
                poster={selected.poster}
                label={`${selected.label} recorded example`}
              />
            ) : (
              <img
                key={selected.task}
                src={selected.image}
                alt={`LibreYOLO ${selected.label.toLowerCase()} output`}
                loading="lazy"
                width="1280"
                height="720"
              />
            )}
          </div>
          <div className={styles.taskInfo}>
            <div className={styles.taskTitle} aria-live="polite">
              <h3>{selected.label}</h3>
              <span>
                {selected.models.length}{" "}
                {selected.models.length === 1 ? "model" : "models"}
              </span>
            </div>
            <p>{descriptions[selected.task] || selected.blurb}</p>
            <label className={styles.modelSearch}>
              <Search size={15} />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Find a model"
                aria-label={`Find a ${selected.label.toLowerCase()} model`}
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear model search"
                >
                  <X size={14} />
                </button>
              )}
            </label>
            <div
              className={styles.modelList}
              aria-label={`${selected.label} models`}
            >
              {models.map((model) => (
                <Link key={model.key} href={model.docsUrl}>
                  <span>{model.name}</span>
                </Link>
              ))}
              {!models.length && (
                <p className={styles.noModels}>
                  No matching model. Try another name.
                </p>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
