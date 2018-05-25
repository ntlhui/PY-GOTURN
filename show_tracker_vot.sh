DEPLOY_PROTO='./nets/tracker.prototxt'		 
CAFFE_MODEL='./nets/tracker_init.caffemodel'		
TEST_DATA_PATH='../vot'		

PYTHONPATH = $PYTHONPATH:./goturn/loader/

python -m goturn.test.show_tracker_vot \
	--p $DEPLOY_PROTO \
	--m $CAFFE_MODEL \
	--i $TEST_DATA_PATH \
	--g 0
