#!/bin/bash
#
# update bibliography from mendeley

echo "Updating phd.bib from Mendeley..."
echo ""
mv backmatter/phd.bib backmatter/phd.bib~
cp $HOME/Documents/bibtex/phd.bib backmatter/phd.bib

# update glossary
makeglossaries main
# update acronyms
