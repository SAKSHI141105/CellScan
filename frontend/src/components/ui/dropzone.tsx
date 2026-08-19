import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

export function Dropzone({
  onFiles,
  accept,
  multiple = false,
  hint,
}: {
  onFiles: (files: File[]) => void;
  accept?: Record<string, string[]>;
  multiple?: boolean;
  hint: string;
}) {
  const onDrop = useCallback((accepted: File[]) => onFiles(accepted), [onFiles]);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, multiple });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-border bg-muted/40 px-6 py-10 text-center transition-colors",
        isDragActive && "border-primary bg-primary/5"
      )}
    >
      <input {...getInputProps()} />
      <UploadCloud className="h-8 w-8 text-muted-foreground" />
      <p className="text-sm font-medium">{isDragActive ? "Drop to upload" : "Drag & drop, or click to browse"}</p>
      <p className="text-xs text-muted-foreground">{hint}</p>
    </div>
  );
}
