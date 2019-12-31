# -*- coding: gb2312 -*-
import os
from flask import Flask, request, flash, url_for, render_template
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename, redirect
import os
from PIL import Image
import numpy as np
import tensorflow as tf


def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data=np.array(im)
    return data

def read_lable(img_name):
    basename = os.path.basename(img_name)
    data = basename.split('_')[0]
    return data
images=[]
lables=[]
#  2.����flaskӦ�ó���ʵ��
#  ��Ҫ����__name__,������Ϊ��ȷ����Դ���ڵ�·��
app = Flask(__name__)
bootstrap = Bootstrap(app)  #bootstrap��ǰ�˿����Ŀ���Ѿ�д�õ���ʽ��Ͷ���
app.config['SECRET_KEY'] = os.urandom(24) #��������չ


@app.route('/', methods=['POST', 'GET'])
def process():
    if request.method == 'POST':
        f = request.files.get('fileupload')
        basepath = os.path.dirname(__file__)
        if f:
            filename = secure_filename(f.filename)
            types = ['jpg', 'png', 'tif']
            if filename.split('.')[-1] in types:#�ж�����
                uploadpath = os.path.join(basepath, 'static/uploads', filename)
                f.save(uploadpath)
                print('uploadpath:',uploadpath)
                # G:\�˹�����\static/uploads\1_1101a.png

                images.append(read_image(uploadpath))
                lables.append(int(read_lable(uploadpath)))

                y_test = np.array(lables)
                x_test = np.array(images)

                model = tf.keras.models.load_model('11.h5')
                predictions = model.predict(x_test)

                acc = []
                ac = []
                for j, i in enumerate(range(len(y_test))):
                    acc.append(int(np.argmax(predictions[i])))
                    ac.append(y_test[j])
                print('��ȷ��', ac)
                print('Ԥ���', acc)
                flash( 'Ԥ���:'+str(acc)+'    ��ȷ��'+ str(ac) , 'success')
            else:
                flash('δ֪����', 'danger')
        else:
            flash('û��ѡ���ļ�', 'danger')

        return redirect(url_for('process'))

    return render_template('base.html')

if __name__ == '__main__':
    app.run(debug=True)
