import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
from tensorflow_addons.metrics import F1Score
from tensorflow_hub import KerasLayer


def ds(name):
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    # class KerasLayer(hub.KerasLayer):
    #     pass

    # 1. 加载模型
    custom_objects = {
        'F1Score': F1Score,
        'Precision': Precision,
        'Recall': Recall
    }
    model = tf.keras.models.load_model(r'C:\Users\Lawaiian\PycharmProjects\dlmi\model\dlmi_3_model.h5',
                                       custom_objects={'KerasLayer': KerasLayer}, compile=False)
    model.compile()

    # 2. 加载图像进行预处理
    img_path = rf'C:\Users\Lawaiian\WebstormProjects\dlmi_frontend\myapp\public\upload\{name}'

    # 3. 进行推断
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    # 4. 输出预测结果
    print(predictions)
    print(len(predictions[0]))

    sorted_indices = sorted(enumerate(predictions[0]), key=lambda x: x[1], reverse=True)

    # 提取前五个元素的索引
    top_five_indices = [index for index, value in sorted_indices[:7]]

    file = open(r"C:\Users\Lawaiian\WebstormProjects\dlmi_frontend\myapp\public\upload\dlmi_result.txt", 'w')
    print("Top-5 probabilities index:", top_five_indices)

    for idx in top_five_indices:
        print(class_names[idx], predictions[0][idx])

    # res = str(cls_names[0]) + " " + str(predictions[0][0])
    if predictions[0][top_five_indices[0]] > 0.8:
        res = str(top_five_indices[0] + 1)
        file.write(res)
    else:
        file.write("8")

    file.close()

    return "Hi"


if __name__ == '__main__':
    ds(sys.argv[1])
