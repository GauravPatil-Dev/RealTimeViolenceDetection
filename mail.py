from flask import Flask, render_template, request, redirect, jsonify
from flask_mail import Mail
from flask_mail import Message

app = Flask(__name__)
app.config.update(dict(
    DEBUG=True,
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USE_SSL=False,
    MAIL_USERNAME='amaljosem7@gmail.com',
    MAIL_PASSWORD='cjhqfnswgztlkvqu',
))
mail = Mail(app)
@app.route("/")
def index():
    msg = Message("Violence Detected", sender="hellomaneeshp@gmail.com",
                  recipients=["mputhenhouse8@gmail.com"])    
    msg.html = "<h3>Real Time Violence Detection System Alert</h3>"
    with app.open_resource("static/plots/fight1.png") as fp:
        msg.attach("fight1.png", "image/png", fp.read())
        mail.send(msg)
    return "Hello World"


if __name__ == "__main__":
    app.run(debug=True)
