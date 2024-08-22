import logging

logging.basicConfig(filename='error.log', level=logging.DEBUG)

def add_attendance(name):
    try:
        username = name.split('_')[0]
        userid = name.split('_')[1]
        current_time = datetime.now().strftime("%I:%M:%S %p")
        df = pd.read_csv(os.path.join('Attendance', f'Attendance-{datetoday}.csv'))
        if int(userid) not in list(df['Roll']):
            with open(os.path.join('Attendance', f'Attendance-{datetoday}.csv'), 'a') as f:
                f.write(f'\n{username},{userid},{current_time}')
    except Exception as e:
        logging.error(f"Error adding attendance for {name}: {e}")

@app.route('/start', methods=['GET'])
def start():
    try:
        if 'face_recognition_model.pkl' not in os.listdir('static'):
            return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
                                   mess='There is no trained model in the static folder. Please add a new face to continue.')

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render_template('home.html', mess="Could not access the webcam.")

        ret = True
        while ret:
            ret, frame = cap.read()
            if extract_faces(frame) != ():
                (x, y, w, h) = extract_faces(frame)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))[0]
                add_attendance(identified_person)
                cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
            cv2.imshow('Attendance', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2)
    except Exception as e:
        logging.error(f"Error in /start route: {e}")
        return render_template('home.html', mess="An error occurred while processing the request.")

@app.route('/add', methods=['GET', 'POST'])
def add():
    try:
        newusername = request.form['newusername']
        newuserid = request.form['newuserid']
        userimagefolder = os.path.join('static', 'faces', f'{newusername}_{str(newuserid)}')
        if not os.path.isdir(userimagefolder):
            os.makedirs(userimagefolder)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return render_template('home.html', mess="Could not access the webcam.")
        i, j = 0, 0
        while 1:
            _, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)
                if j % 10 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y + h, x:x + w])
                    i += 1
                j += 1
            if j == 500:
                break
            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        train_model()
        names, rolls, times, l = extract_attendance()
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2)
    except Exception as e:
        logging.error(f"Error in /add route: {e}")
        return render_template('home.html', mess="An error occurred while adding the new user.")

if __name__ == '__main__':
    app.run(debug=True)
