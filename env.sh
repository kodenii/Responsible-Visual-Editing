
source activate rve
pip install git+https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git@package
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
cd ops && sh make.sh && cd ..