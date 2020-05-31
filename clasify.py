import time
import os
import sys
import cv2
import numpy as np
from data import DataSet
from extractor import Extractor
from keras.models import load_model
from insert1 import insert_img
from Connector_mysql import connect
import winsound
import re
import math
import  io
print("Loading Model .......")
saved_LSTM_model = load_model("data\\checkpoints\\lstm-features.022-0.035.hdf5",compile='False')
extract_model = Extractor(image_shape=(320,240, 3))
print("****************************Model Ready.......***************************")

def video(video_file):
    #print('time take to load imports {:0.3f}'.format(time.time() - start))
    start = time.time()
    '''print(sys.argv)
    if (len(sys.argv) == 2):
        #seq_length = int(sys.argv[1])
        #class_limit = int(sys.argv[2])
        #saved_model = sys.argv[3]
        #video_file = sys.argv[1]
    else:
        print ("Usage: python clasify.py video_file_name")
        print ("Example: python clasify.py some_video.mp4")
        exit (1)
    '''
    file_path = re.compile(r'[^\\/:*?"<>|\r\n]+$')
    file_name = file_path.search(video_file)
    start  = None
    f_name =  file_name.group()[start:-4]
    capture = cv2.VideoCapture(video_file)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    seq_length = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(video_file)
    video_writer = cv2.VideoWriter( f_name+ "_result.mp4", fourcc, 15, (width,height))
    # Get the dataset.
    data = DataSet(seq_length=41, class_limit=2, image_shape=(240,320, 3))
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    # get the model.
    #start = time.time()
    
    #print("Loading Model .......")
    #extract_model = Extractor(image_shape=(240,320, 3))
    #saved_LSTM_model = load_model("F:\\BE Project work\\LSTM-video-classification-master\\data\\checkpoints\\lstm-features.007-0.264.hdf5",compile='False')
    #print(capture)
    #print("Captured Video....")
    #print("Model Loaded.......")
    #print('time required to load model{:0.3f}'.format(time.time() - start))
    cam_id=1002
    frames = []
    frame_count = 0
    try:
        conn = connect()
    except:
        print("DATABASE Error")

    start = time.time()
    frameRate = capture.get(5) #frame rate

    while True:
        r1, frame1 = capture.read()
        ret, frame = capture.read()
        #frameId = capture.get(1) #current frame number
        #print(capture.get(1))
        # Bail out when the video file ends
        if not ret:
            break
        # Save each frame of the video to a list
        
        frame_count += 1
        image1 = cv2.resize(frame, (240, 320))
        frames.append(image1)
       
        #print("LINE 86")
        
        #print("LINE 89")
        if frame_count < seq_length:
            continue # capture frames untill you get the required number for sequence
        else:
            frame_count = 0

        # For each frame extract feature and prepare it for classification
        sequence = []
        for image in frames:
            features = extract_model.extract_image(image)
            #print(features)
            sequence.append(features)
          
        # Clasify sequence
        prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
        values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
        #print(values)

        #for i in range(len(prediction)):
        if prediction.item(1) >= 0.9:
                insert_img(conn,frame,cam_id,'high')
                winsound.Beep(2500, 100)
        elif prediction.item(1) >= 0.75:
                insert_img(conn,frame,cam_id,'medium')
                winsound.Beep(1000, 50)
        elif prediction.item(1) >= 0.65:
                insert_img(conn,frame,cam_id,'low')  
                winsound.Beep(500, 25)

        #else:   
        #        print ("value is too high")
        # Add prediction to frames and write them to new video
        for image in frames:
            for i in range(len(values)):
                cv2.putText(image, values[i], (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), lineType=cv2.LINE_AA)
            #cv2.imshow("Frame",image)
            image = cv2.resize(image, (height, width))
           
            f1=image.copy()
            encode_return_code, image_buffer = cv2.imencode('.jpg', f1)
            io_buf = io.BytesIO(image_buffer)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')
            video_writer.write(image)

            if  cv2.waitKey(1) and 0xFF == ord('q'):
               break

        frames = []
    cv2.destroyAllWindows()
    print('time required {:0.3f}'.format(time.time() - start))
    conn.close()
    video_writer.release()
    #return f_name + "_result.mp4"    
def video_live():
    #print('time take to load imports {:0.3f}'.format(time.time() - start))
    start = time.time()
    '''print(sys.argv)
    if (len(sys.argv) == 2):
        #seq_length = int(sys.argv[1])
        #class_limit = int(sys.argv[2])
        #saved_model = sys.argv[3]
        #video_file = sys.argv[1]
    else:
        print ("Usage: python clasify.py video_file_name")
        print ("Example: python clasify.py some_video.mp4")
        exit (1)
    '''
    #file_path = re.compile(r'[^\\/:*?"<>|\r\n]+$')
    #file_name = file_path.search(video_file)
    #start  = None
    #f_name =  file_name.group()[start:-4]
    if os.environ.get('OPENCV_CAMERA_SOURCE'):
        capture = cv2.VideoCapture(int(os.environ['OPENCV_CAMERA_SOURCE']))
    else:
        capture = cv2.VideoCapture(0)

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) )  # float
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    seq_length = 5
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #print(f_name)
    #video_writer = cv2.VideoWriter(f_name + "_result.mp4", fourcc, 15, (320,240))
    # Get the dataset.
    data = DataSet(seq_length=41, class_limit=2, image_shape=(360,640, 3))
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

    # get the model.
    #start = time.time()
    
    #print("Loading Model .......")
    #extract_model = Extractor(image_shape=(240,320, 3))
    #saved_LSTM_model = load_model("F:\\BE Project work\\LSTM-video-classification-master\\data\\checkpoints\\lstm-features.007-0.264.hdf5",compile='False')
    #print(capture)
    #print("Captured Video....")
    #print("Model Loaded.......")
    #print('time required to load model{:0.3f}'.format(time.time() - start))
    cam_id=1002
    frames = []
    frame_count = 0
    try:
        conn = connect()
    except:
        print("DATABASE Error")

    while True:
        r1, frame1 = capture.read()
        ret, frame = capture.read()
        #frameId = capture.get(1) #current frame number
        #print(capture.get(1))
        # Bail out when the video file ends
        if not ret:
            break
        # Save each frame of the video to a list
        
        frame_count += 1
        image1 = cv2.resize(frame, (240, 320))
        frames.append(image1)

        if frame_count < seq_length:
            continue # capture frames untill you get the required number for sequence
        else:
            frame_count = 0

        # For each frame extract feature and prepare it for classification
        sequence = []
        for image in frames:
            features = extract_model.extract_image(image)
            #print(features)
            sequence.append(features)
          
        # Clasify sequence
        prediction = saved_LSTM_model.predict(np.expand_dims(sequence, axis=0))
        values = data.print_class_from_prediction(np.squeeze(prediction, axis=0))
        #print(values)

        #for i in range(len(prediction)):
        if prediction.item(1) >= 0.9:
                insert_img(conn,frame,cam_id,'high')
                winsound.Beep(2500, 100)
        elif prediction.item(1) >= 0.75:
                insert_img(conn,frame,cam_id,'medium')
                winsound.Beep(1000, 50)
        elif prediction.item(1) >= 0.65:
                insert_img(conn,frame,cam_id,'low')  
                winsound.Beep(500, 25)

        #else:   
        #        print ("value is too high")
        # Add prediction to frames and write them to new video
        for image in frames:
            for i in range(len(values)):
                cv2.putText(image, values[i], (40, 40 * i + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), lineType=cv2.LINE_AA)
            #cv2.imshow("Frame",image)
            #video_writer.write(image)
            image = cv2.resize(image, (height, width))
            f1=image.copy()
            encode_return_code, image_buffer = cv2.imencode('.jpg', f1)
            io_buf = io.BytesIO(image_buffer)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')
            if  cv2.waitKey(1) and 0xFF == ord('q'):
               break

        frames = []
    if 0xFF == ord('q'):
        exit(1)    
    cv2.destroyAllWindows()
    #print('time required {:0.3f}'.format(time.time() - start))
    conn.close()
    #video_writer.release()
