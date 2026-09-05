import RESULTS from './results-summary.json'

const ap = (value) => (value * 100).toFixed(2)
const archive = 'https://huggingface.co/datasets/LibreYOLO/rf100-vl-results'

export default function RF100VLResults() {
  return (
    <section aria-label="RF100-VL benchmark results" className="not-prose my-8" style={{ width: 'min(94vw, 1040px)', marginLeft: 'calc(50% - min(47vw, 520px))' }}>
      <div className="overflow-x-auto rounded-xl border border-surface-200 dark:border-surface-700 bg-white dark:bg-surface-900" tabIndex={0} role="region" aria-label="RF100-VL results table">
        <table className="w-full md:min-w-[760px] text-sm border-collapse">
          <caption className="px-5 py-4 text-left border-b border-surface-200 dark:border-surface-700 text-surface-600 dark:text-surface-300">
            <strong className="text-surface-900 dark:text-white">17 configurations, 100 test splits each.</strong> Sorted by mean AP50:95.
          </caption>
          <thead>
            <tr className="text-xs text-surface-600 dark:text-surface-400 bg-surface-50 dark:bg-surface-800/50">
              <th scope="col" className="text-left px-5 py-3">Configuration</th>
              <th scope="col" className="hidden md:table-cell text-right px-3 py-3">Pixels</th>
              <th scope="col" className="hidden md:table-cell text-right px-3 py-3">AP50</th>
              <th scope="col" className="text-right md:text-left px-5 md:px-3 py-3 md:w-52">AP50:95</th>
              <th scope="col" className="hidden md:table-cell text-right px-3 py-3">Median train</th>
              <th scope="col" className="hidden md:table-cell text-right px-5 py-3">Records</th>
            </tr>
          </thead>
          <tbody>
            {RESULTS.map((row) => (
              <tr key={row.id} className="border-t border-surface-100 dark:border-surface-800 hover:bg-surface-50 dark:hover:bg-surface-800/40">
                <th scope="row" className="px-5 py-3 text-left font-medium text-surface-800 dark:text-surface-100">
                  {row.model}-{row.size}
                  {row.id.startsWith('yolov9') && <span className="block text-[11px] font-normal text-surface-500 dark:text-surface-400">Before training fixes</span>}
                  <span className="block md:hidden mt-1 text-xs font-normal text-surface-500 dark:text-surface-400">{row.inputSize}px · {row.trainMin.toFixed(1)} min median</span>
                  <span className="flex md:hidden gap-3 mt-1.5 text-xs font-normal">
                    <a className="text-libre-700 dark:text-libre-400 underline underline-offset-2" href={`${archive}/blob/main/${row.submissionPath}`} target="_blank" rel="noopener noreferrer" aria-label={`Read ${row.model}-${row.size} submission`}>Result</a>
                    <a className="text-libre-700 dark:text-libre-400 underline underline-offset-2" href={`${archive}/tree/main/${row.runPath}`} target="_blank" rel="noopener noreferrer" aria-label={`Browse ${row.model}-${row.size} run artifacts`}>Run artifacts</a>
                  </span>
                </th>
                <td className="hidden md:table-cell px-3 py-3 text-right font-mono tabular-nums text-surface-600 dark:text-surface-400">{row.inputSize}</td>
                <td className="hidden md:table-cell px-3 py-3 text-right font-mono tabular-nums text-surface-600 dark:text-surface-400">{ap(row.map50)}</td>
                <td className="px-5 md:px-3 py-3">
                  <div className="flex items-center justify-end md:justify-start gap-3">
                    <span className="hidden md:block h-1.5 w-24 bg-surface-100 dark:bg-surface-800 rounded-full overflow-hidden" aria-hidden="true">
                      <span className="block h-full bg-libre-500 rounded-full" style={{ width: `${row.map * 100}%` }} />
                    </span>
                    <span className="font-mono font-semibold tabular-nums text-surface-900 dark:text-white">{ap(row.map)}</span>
                  </div>
                </td>
                <td className="hidden md:table-cell px-3 py-3 text-right font-mono tabular-nums text-surface-600 dark:text-surface-400">{row.trainMin.toFixed(1)} min</td>
                <td className="hidden md:table-cell px-5 py-3 text-right">
                  <a className="text-libre-700 dark:text-libre-400 underline underline-offset-2" href={`${archive}/blob/main/${row.submissionPath}`} target="_blank" rel="noopener noreferrer" aria-label={`Read ${row.model}-${row.size} submission`}>Result</a>
                  <span className="text-surface-300 dark:text-surface-600"> / </span>
                  <a className="text-libre-700 dark:text-libre-400 underline underline-offset-2" href={`${archive}/tree/main/${row.runPath}`} target="_blank" rel="noopener noreferrer" aria-label={`Browse ${row.model}-${row.size} run artifacts`}>Run</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs leading-relaxed text-surface-600 dark:text-surface-400">
        Bars use a fixed 0 to 100 AP scale. Training times are recorded campaign medians, not controlled speed comparisons.
        Code versions, job sharing and recovery differ across campaigns. The methodology below describes those limits.
      </p>
      <p className="mt-2 text-xs leading-relaxed text-surface-600 dark:text-surface-400">
        YOLO-NAS starts from Deci weights under its separate{' '}
        <a className="text-libre-700 dark:text-libre-400 underline" href="https://github.com/Deci-AI/super-gradients/blob/master/LICENSE.YOLONAS.md" target="_blank" rel="noopener noreferrer">non-commercial weights license</a>.
        RF-DETR, EdgeCrafter and YOLOX use Apache-2.0 weights; YOLOv9 uses MIT weights.
      </p>
    </section>
  )
}
