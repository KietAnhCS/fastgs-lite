# latexmk config for this folder. Run `latexmk` (no flags needed) or `latexmk -pvc`.

# pdflatex only: this CV has no bibliography, so bibtex/biber passes are wasted
# work. latexmk still reruns pdflatex on its own if labels or the .out file
# change, so cross-references stay correct.
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';

# Keep the 6 build artifacts (.aux .log .fls .fdb_latexmk .out .synctex.gz) out
# of the OneDrive-synced source folder -- every compile rewrote them and kicked
# off a sync. The PDF lands in out/ too.
$out_dir = 'out';

# Under -pvc, a compile error should leave the watcher running so the next save
# retries, instead of dropping back to the shell.
$pvc_view_file_update = 1;
$preview_continuous_mode = 0;

# Open the PDF with whatever Windows has registered for .pdf.
$pdf_previewer = 'start %O %S';

# `latexmk -c` removes these too.
$clean_ext = 'synctex.gz run.xml bbl';
