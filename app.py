from flask import Flask, render_template_string
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    # Load accuracy from your model or set a static value
    accuracy = 5.00  # Replace with your actual value or load dynamically
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f0f4f8; margin: 0; padding: 0; }
            .container { max-width: 500px; margin: 80px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ccc; padding: 40px; text-align: center; }
            h1 { color: #2d7a2d; }
            .accuracy { font-size: 2em; color: #333; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Plant Disease Classifier</h1>
            <div class="accuracy">Validation Accuracy: {{ accuracy }}%</div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, accuracy=accuracy)

if __name__ == "__main__":
    app.run(debug=True)