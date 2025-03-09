import os
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone, timedelta

app = Flask(__name__)

# Obtener la URL de la base de datos desde Render
url_internal = os.getenv("DATABASE_URL", "postgresql://flask_postgres_j4of_user:kWTVfzGKg8QNGXMhU6GtnY9H8cCflqpc@dpg-cv6tjadumphs738d9nf0-a/flask_postgres_j4of")

# Reemplazar 'postgres://' por 'postgresql://' (Render usa un formato diferente)
if url_internal.startswith("postgres://"):
    url_internal = url_internal.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = url_internal
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Definir el modelo
class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200))
    done = db.Column(db.Boolean)
    fecha_creacion = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc) + timedelta(hours=-5))

# Crear la tabla en la base de datos
with app.app_context():
    db.create_all()

@app.route('/')
def home():
    tasks = Task.query.all()
    dias_semana = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    return render_template('index.html', tasks=tasks, dias_semana=dias_semana)

@app.route('/create-task', methods=['POST'])
def create():
    task = Task(content=request.form['content'], done=False)
    db.session.add(task)
    db.session.commit()
    return redirect(url_for('home'))

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    task = Task.query.get(id)
    if request.method == 'POST' and task:
        task.content = request.form['content-edit']
        db.session.commit()
        return redirect(url_for('home'))
    return render_template('edit.html', task=task)

@app.route('/done/<int:id>')
def done(id):
    task = Task.query.get(id)
    if task:
        task.done = not task.done
        db.session.commit()
    return redirect(url_for('home'))

@app.route('/delete/<int:id>')
def delete(id):
    task = Task.query.get(id)
    if task:
        db.session.delete(task)
        db.session.commit()
    return redirect(url_for('home'))
