all: ms.pdf

%.pdf: %.tex
	tectonic ms.tex
	#cp ms.tex final
	rm -rf *Notes.bib
	rm -rf *.log
	rm -rf *.bbl
	rm -rf *.aux
	rm -rf *.out
	rm -rf *.blg
	rm -rf *.dvi
	# rm ms.fdb_latexmk
	# rm ms.fls
	# rm ms.synctex.gz
	mv ms.pdf final/
	open final/ms.pdf
