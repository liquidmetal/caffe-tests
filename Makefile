FILES = usinha-report-assignment1.pdf
SEC_4 = Sec_4/training.log Sec_4/train_val.prototxt Sec_4/solver.prototxt Sec_4/cls_results*.txt
SEC_5 = Sec_5/training.log Sec_5/train_val.prototxt Sec_5/solver.prototxt Sec_5/cls_results*.txt
SEC_6 = Sec_6/train_val.prototxt Sec_6/solver.prototxt Sec_6/test_bbox.py Sec_6/bbox-result*.txt
SEC_7 = Sec_7/train_val.prototxt Sec_7/training.log Sec_7/label-bbox-*.txt
SEC_8 = Sec_8/generate-matches.py Sec_8/*.prototxt Sec_8/copier.py
SEC_9 = Sec_9/drop_output_layer.cpp Sec_9/drop_output_layer.cu Sec_9/drop_output_layer.hpp Sec_9/training.log Sec_9/ec-label-bbox-* Sec_9/test_label_bbox.py Sec_9/*.prototxt
all:
	cp writeup.pdf usinha-report-assignment1.pdf
	zip -r usinha.zip ${FILES} ${SEC_4} ${SEC_5} ${SEC_6} ${SEC_7} ${SEC_8} ${SEC_9}
	rm usinha-report-assignment1.pdf
