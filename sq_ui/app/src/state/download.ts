/** Keep blob URLs alive until the browser has had time to start the download. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  // Revoking immediately can cancel downloads in WebKit-based browsers.
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
