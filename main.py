#main.py
# python main.py --conf conf.json
# python main.py --conf conf.json --video videos/example_01.mp4

# Importing the necessary packages
from modules.tempimage import TempImage
# from picamera.array import PiRGBArray
# from picamera import PiCamera
from imutils.video import VideoStream
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")

args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
	print("Camera started...")
	time.sleep(2.0)
	# avg = None
	lastUploaded = datetime.datetime.now()
	motionCounter = 0

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
firstFrame = None


# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print("Successfully linked to dropbox")


# loop over the frames of the video
while True:
	frame = vs.read()  
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
	timestamp = datetime.datetime.now()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	frameDelta = cv2.absdiff(firstFrame, gray)

	# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]

	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"

	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
    
	# check to see if the room is occupied
	if text == "Occupied":
		lastUploaded = datetime.datetime.now()
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1

			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				# check to see if dropbox sohuld be used
				if conf["use_dropbox"]:
					# write the image to temporary file
					t = TempImage()
					cv2.imwrite(t.path, frame)

					# upload the image to Dropbox and cleanup the tempory image
					print("[Uploaded to dropbox] {}".format(ts))
					path = "/{base_path}/{timestamp}.jpg".format(
					    base_path=conf["dropbox_base_path"], timestamp=ts)
					client.files_upload(open(t.path, "rb").read(), path)
					t.cleanup()

				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0

	# otherwise, the room is not occupied
	else:
		motionCounter = 0

	# check to see if the frames should be displayed to screen
	if conf["show_video"]:
		# display the security feed
		cv2.imshow("Surveillance Feed: Hye!! you are under my watch", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break

	
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
