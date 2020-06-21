from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, BooleanField, SelectField,\
    SubmitField, MultipleFileField
from wtforms.validators import DataRequired, Length, Email, Regexp
from wtforms import ValidationError
from flask_wtf.file import FileAllowed, FileRequired, FileField



class UploadForm(FlaskForm):
    name = StringField('Provider Name', validators=[DataRequired()])
    title = StringField('Title')
    email = StringField('Email', validators=[DataRequired(), Email()])
    # validated = BooleanField('Validated Data')
    comments = TextAreaField("Comments")
    # language = SelectField('More choices', choices = [('cpp', 'c1'), ('py', 'c2')])

    data_file = FileField('X-ray image File', validators=[
                                    FileRequired(),
                                    # FileAllowed(['jpeg', 'jpg', 'png', 'bmp'],
                                    #             'Only jpeg, jpg, png, or bmp images are allowed!')
                ])

    submit = SubmitField('Get Recommendation', render_kw={'class': 'btn btn-primary pull-xs-right tm-button tm-button-normal'})
