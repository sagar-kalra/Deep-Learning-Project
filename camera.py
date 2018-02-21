def capture():
    import cv2
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Capture")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("Capture", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = 'captured_sign.png'
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break

    cam.release()

    cv2.destroyAllWindows()
