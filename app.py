
from flask import Flask,render_template,request
from text_summary import summarizer
from gen import generate_contract
from transformers import pipeline


app= Flask(__name__)
contract_generator = pipeline("text-generation", model="gpt2")

@app.route('/')
def index():
    return render_template("button.html")

@app.route('/run_index', methods=['POST'])
def run_index():
    return render_template("index.html")
@app.route('/run_co', methods=['POST'])
def run_co():
    return render_template("co.html")

@app.route('/analyse',methods=["GET","POST"])
def analyse():
    if request.method=="POST":
        rawtext = request.form['rawtext']
        summary,original_text,len_orig,len_summary=summarizer(rawtext)

    return render_template('summary.html',summary=summary, original_text=original_text,len_orig=len_orig,len_summary=len_summary)

@app.route('/gen_c', methods=["GET","POST"])
def gen_c():
    if request.method=="POST":
        prompt = request.form.get('prompt') 
        prompt= str(prompt)
        generated_contract = generate_contract(prompt)
    
    return render_template('gen.html', prompt=prompt, generated_contract=generated_contract)

@app.route('/contract', methods=["GET","POST"])
def contract():
    if request.method=="POST":
        party_a_name = request.form.get('partyA')
        party_b_name = request.form.get('partyB')
        contract_type = request.form.get('contractType')
        contract_details = request.form.get('contractDetails')

        
        prompt = (f'''Party A: {party_a_name}<br>
                   Party B: {party_b_name}<br>
                   Contract Type: {contract_type}<br>
                   Details: {contract_details}<br>''')
        contract = contract_generator(prompt, max_length=100, num_return_sequences=1)
        contract_text = contract[0]['generated_text']
    return render_template('co.html', contract=contract_text)
        

if __name__ == '__main__':
    app.run(debug=True)