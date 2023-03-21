from flask import Flask,render_template,request
from app.utils import prediction
import CONFIG

app=Flask(__name__)


@app.route("/")
def fun():
    return render_template("index.html")

@app.route("/data",methods=["POST"])
def guess():
    input_data=request.form
    obj=prediction()
    outcome=obj.class_predict(input_data)
    print(outcome)

    if outcome[0]==0:
        result=0
    elif outcome[0]==1:
        result=1

    return render_template("index.html",predicted_class=result)



if __name__=="__main__":
    app.run(host=CONFIG.host,port=CONFIG.port)
