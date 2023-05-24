from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()  # 获取前端发送的 JSON 数据
    num1 = data['num1']
    num2 = data['num2']
    operation = data['operation']

    if operation == 'add':
        result = num1 + num2
    elif operation == 'subtract':
        result = num1 - num2
    elif operation == 'multiply':
        result = num1 * num2
    elif operation == 'divide':
        result = num1 / num2
    else:
        return jsonify({'error': 'Invalid operation'})

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()
