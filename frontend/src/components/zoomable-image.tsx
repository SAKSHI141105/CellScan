import { useState } from "react";
import { createPortal } from "react-dom";
import { Maximize2, X } from "lucide-react";

/** Large image card with a click-to-fullscreen lightbox — used in the
 * enlarged Raw/CLAHE/Grad-CAM inspection row. Kept as a plain portal overlay
 * rather than pulling in a dialog library for one interaction. */
export function ZoomableImage({ src, label, caption }: { src: string; label: string; caption?: string }) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <figure className="group relative overflow-hidden rounded-xl border border-border bg-muted/30">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="block w-full cursor-zoom-in"
          aria-label={`Zoom ${label}`}
        >
          <img src={src} className="aspect-square w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]" />
          <span className="absolute right-2 top-2 flex items-center gap-1 rounded-md bg-black/60 px-2 py-1 text-[10px] font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
            <Maximize2 className="h-3 w-3" /> Inspect
          </span>
        </button>
        <figcaption className="border-t border-border bg-card px-3 py-2 text-xs font-semibold">
          {label}
          {caption && <span className="ml-1.5 font-normal text-muted-foreground">{caption}</span>}
        </figcaption>
      </figure>

      {open &&
        createPortal(
          <div
            className="fixed inset-0 z-[100] flex items-center justify-center bg-black/85 p-4 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          >
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="absolute right-4 top-4 rounded-full bg-white/10 p-2 text-white hover:bg-white/20"
              aria-label="Close"
            >
              <X className="h-5 w-5" />
            </button>
            <img
              src={src}
              className="max-h-[88vh] max-w-[92vw] rounded-lg object-contain shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
            <span className="absolute bottom-6 left-1/2 -translate-x-1/2 rounded-md bg-black/60 px-3 py-1.5 text-xs font-medium text-white">
              {label}
            </span>
          </div>,
          document.body
        )}
    </>
  );
}
