from flask import Flask, request
import logging
from werkzeug.exceptions import BadRequest
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
import json


app = Flask(__name__)
#logger= logging.getLogger(__name__)

#-------Issue lies in log records working on localhost but not generated after deployment----
#-------------------------------------To be fixed-------------------------------

#ch= logging.StreamHandler()
#fh= logging.FileHandler('lsq-ml-pk-logger.log')
#ch.setLevel(logging.DEBUG)
#fh.setLevel(logging.DEBUG)

#c_format= logging.Formatter('%(levelname)s : %(name)s : %(message)s')
#f_format= logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s',  datefmt='%Y-%m-%d %H:%M:%S')
#ch.setFormatter(c_format)
#fh.setFormatter(f_format)

#app.logger.addHandler(ch)
#app.logger.addHandler(fh)


#app.logger.debug('This is a DEBUG message')
#app.logger.info('This is an INFO message')
#app.logger.warning('Be Careful, This is a warning')
#app.logger.error('This is an error log')
#app.logger.critical('Fatal, a critical message')




model_path= 'model/lendsqrdatapredmode.pkl'
model= pickle.load(open('model/lendsqrdatapredmode.pkl', 'rb'))
le= LabelEncoder()



@app.route('/', methods=['POST', 'GET'])
def home():
	#app.logger.error('Preprocessing...This is the home page')
	return_data= {
		'error': "0",
		'message': "Successful"
	}
	return app.response_class(response= json.dumps(return_data), mimetype='application/json')


#-----------------------------------------------Error Messages-----------------------------------------------------------
lm_err= 'Loan Amount should be a number or be between 100 and 9999999'
id_err= 'Interest Due should be a number or be between 0 and 999999'
cs_err= 'Credit Score should be a number or be between 0 and 100'
ir_err= 'Interest Rate should be a number or be between 0 and 100'
tn_err= "Tenor should be an object value in this format ('7 days'), not a number"
ppd_err= "Proposed Payday should be an object value in this format ('7 days'), not a number"
ra_err= 'Requested Amount should be a number or be between 0 and 999999999'
fr_err= 'Failed Request should be a number or be between 0 and 99999'
l_err= 'Logins should be a number or be between 0 and 99999'
pr_err= 'Passed Request should be a number or be between 0 and 999'
ar_err= 'All Request should be a number or be between o and 999'
pn_err= 'Phone Number should be a number or be between 1 and 99'
em_err= 'Emails should be a number or be between 1 and 99'
ld_err= 'Lenders should be a number or be between 1 and 99'
ll_err= 'Lending Lenders should be a number or be between 1 and 99'
ln_err= 'Loans should be a number or be between 1 and 99'
dob_err= 'Date of birth should be a date'
wsd_err= 'Work start date should be a date'
la_err= 'Last account should be a date'
fa_err= 'First account should be a date'
cn_err= 'Card Network should be Mastercard, Verve, or Visa'
tr_err= 'Tier should be Tier 1, Tier 2, or Tier 3'
ms_err= 'Marital Status should be Married, Divorced, Single or Widowed'
sic_err= 'Selfie ID Check should be Pending, Successful or Failed'
wev_err= 'Work Email Validated should be a number between 0 or 1'
mni_err= 'Monthly Net Income should be an object value in this format(55,000-100,000), not a number'
bk_err= 'Bank should be a bank name, not a number'
soe_err= 'Sector Of Employment should not be a number'
pp_err= 'Purpose sould not be a number'
lc_err= 'Location should not be a number'
es_err= 'Employment Status should not be a number'
ea_err= 'Educational Attainment should not be a number'
nod_err= 'No of Dependent should be a number'
ph_err= 'Phone Network should be a number'
cd_err= 'Card Expiry should be a 5 digits number or more'
ad_err= 'Address should not be a number'





@app.route('/v1/predict', methods=['POST', 'GET'])
def Prediction():
	try:
		data= request.json
		if data != None:
			#Data Preprocessing
			#app.logger.error('Preprocessing...Please wait, as Prediction is being made on your provided data')
			data= pd.DataFrame(data, index= [1])

			#---------------------------For the loan amount column---------------------------------
			try:
				if data['loan_amount'].values < 100 or data['loan_amount'].values > 9999999:
					#app.logger.error(lm_err)
					return_data= { 
						'error': '3',
						'message': str(lm_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['loan_amount']= float(data['loan_amount'])

			except Exception as e:
				#app.logger.error(lm_err)
				return_data= {
					"error": '3',
					"message": str(lm_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')


			loan_amount= data['loan_amount']

			#-------------------------------------Interest Due Column ----------------------------------------------


			try:
				if data['interest_due'].values < 0 or data['interest_due'].values > 999999:
						#app.logger.error(id_err)
						return_data= { 
							'error': '3',
							'message': str(id_err)
						}
						return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['interest_due']= float(data['interest_due'])

			except Exception as e:
				#app.logger.error(id_err)
				return_data= {
					"error": '3',
					"message": str(id_err)				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			interest_due= data['interest_due']

			#--------------------------------------Interest Rate Column--------------------------------------

			try:
				if data['interest_rate'].values < 0 or data['interest_rate'].values > 100:
					#app.logger.error(ir_err)
					return_data= { 
						'error': '3',
						'message': str(ir_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['interest_rate']= float(data['interest_rate'])
			except Exception as e:
					#app.logger.error(ir_err)
					return_data= {
						"error": '3',
						"message": str(ir_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			interest_rate= data['interest_rate']

			#---------------------------------Credit Score Column---------------------------------------------

			try:
				if data['credit_score'].values < 0 or data['credit_score'].values > 100:
					#app.logger.error(cs_err)
					return_data= { 
						'error': '3',
						'message': str(cs_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['credit_score']= float(data['credit_score'])

			except Exception as e:
				#app.logger.error(cs_err)
				return_data= {
					"error": '3',
					"message": str(cs_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			credit_score = data['credit_score']

			#----------------------------  Tenor Column-----------------------------------------

			try:
				if (data['tenor'].dtype == np.float64 or data['tenor'].dtype == np.int64):
					#app.logger.error(tn_err)
					return_data= { 
						'error': '3',
						'message': str(tn_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['tenor'] = data.tenor.apply(lambda x: str(int(x.split(' ')[0]) * 30) + ' days' if x.endswith('months') else x)
					data['tenor']= data.tenor.apply(lambda x: str(int(x.split(' ')[0]) * 7) + ' days' if x.endswith('weeks') else x)
					data['tenor']= data['tenor'].map(lambda x: x.split(' ')[0]).astype(str)

			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(tn_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			tenor= data['tenor']

			#------------------------------------Proposed Payday Column---------------------------------

			try:
				if (data['proposed_payday'].dtype == np.float64 or data['proposed_payday'].dtype == np.int64):
					#app.logger.error(ppd_err)
					return_data= { 
						'error': '3',
						'message': str(ppd_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['proposed_payday'] = data.proposed_payday.apply(lambda x: str(int(x.split(' ')[0]) * 30) + ' days' if x.endswith('months') else x)
					data['proposed_payday']= data.proposed_payday.apply(lambda x: str(int(x.split(' ')[0]) * 7) + ' days' if x.endswith('weeks') else x)
					data['proposed_payday']= data['proposed_payday'].map(lambda x: x.split(' ')[0]).astype(str)
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(ppd_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			proposed_payday= data['proposed_payday']

			#----------------------------------Requested Amount column ----------------------------------

			try:

				if data['requested_amount'].values < 0 or data['requested_amount'].values > 999999999:
						#app.logger.error(ra_err)
						return_data= { 
							'error': '3',
							'message': str(ra_err)
						}
						return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['requested_amount']= float(data['requested_amount'])

			except Exception as e:
				#app.logger.error(ra_err)
				return_data= {
					"error": '3',
					"message":str(ra_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			requested_amount= data['requested_amount']


			#-------------------------------Failed Request Column------------------------------------
			try:
				if data['failed_requests'].values < 0 or data['failed_requests'].values >  99999:
					#app.logger.error(fr_err)
					return_data= { 
						'error': '3',
						'message': str(fr_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['failed_requests']= int(data['failed_requests'])
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(fr_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			failed_requests= data['failed_requests']


			#-------------------------------Logins Column-------------------------------------------------

			try:
				if data['logins'].values < 0 or data['logins'].values > 99999:
					#app.logger.error(l_err)
					return_data= { 
						'error': '3',
						'message': str(l_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['logins']= int(data['logins'])
					
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(l_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			logins= data['logins']
			


			#----------------------------Passed Request Column-------------------------------

			try:
				if data['passed_requests'].values < 0 or data['passed_requests'].values > 999:
					#app.logger.error(pr_err)
					return_data= { 
						'error': '3',
						'message': str(pr_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['passed_requests']= int(data['passed_requests'])
			except Exception as e:
					return_data= {
						"error": '3',
						"message": str(pr_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			passed_requests= data['passed_requests']

			#---------------------------------All Request Column --------------------------------------------

			try:
				if data['all_requests'].values < 0 or data['all_requests'].values > 999:
					#app.logger.error(ar_err)
					return_data= { 
						'error': '3',
						'message': str(ar_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['all_requests']= int(data['all_requests'])
					
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(ar_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			all_requests= data['all_requests']
		


			#-----------------------------Phone numbers column----------------------------------------

			try:
				if data['phone_numbers'].values < 1 or data['phone_numbers'].values >  99:
					#app.logger.error(pn_err)
					return_data= { 
						'error': '3',
						'message': str(pn_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['phone_numbers']= float(data['phone_numbers'])
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(pn_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')	
			phone_numbers= data['phone_numbers']


			#-----------------------------------Emails Column -----------------------------------------

			try:
				if data['emails'].values < 1 or data['emails'].values > 99:
					#app.logger.error(em_err)
					return_data= { 
						'error': '3',
						'message': str(em_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['emails']= float(data['emails'])
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(em_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			emails= data['emails']

			#--------------------------------Lenders Column------------------------------

			try:
				if data['lenders'].values < 1 or data['lenders'].values > 99:
					#app.logger.error(ld_err)
					return_data= { 
						'error': '3',
						'message': str(ld_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['lenders']= float(data['lenders'])
					
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(ld_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			lenders= data['lenders']

			#----------------------------------------Lending lender and loan columns---------------------------------------

			try:
				if data['lending_lenders'].values < 0 or data['lending_lenders'].values >  99:
					#app.logger.error(ll_err)
					return_data= { 
						'error': '3',
						'message': str(ll_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['lending_lenders']= int(data['lending_lenders'])
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(ll_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			lending_lenders= data['lending_lenders']		


			#--------------------------------------Loans Column--------------------------------

			try:
				if data['loans'].values < 0 or data['loans'].values > 99:
					#app.logger.error(ln_err)
					return_data= { 
						'error': '3',
						'message': str(ln_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['loans']= int(data['loans'])
			except Exception as e:
				return_data= {
					"error": '3',
					"message": str(ln_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			loans= data['loans']

			#--------------------------------The Date Related Columns----------------------------------------


			try:
				if (data['date_of_birth'].dtype == np.float64 or data['date_of_birth'].dtype == np.int64):
					#app.logger.error(dob_err)
					return_data= { 
						'error': '3',
						'message': str(dob_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['date_of_birth']= pd.to_datetime(data['date_of_birth'], errors= "ignore")
					date_of_birth_year= data['date_of_birth'].dt.year
			except Exception as e:
				return_data= {
					'error': '3',
					'message': str(dob_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			date_of_birth_year= date_of_birth_year


			#----------------------------------Work Start Date Column---------------------------------

			try:
				if (data['work_start_date'].dtype == np.float64 or data['work_start_date'].dtype == np.int64):
					#app.logger.error(wsd_err)
					return_data= { 
						'error': '3',
						'message': str(wsd_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['work_start_date']= pd.to_datetime(data['work_start_date'], errors= "ignore")
					work_start_date_year= data['work_start_date'].dt.year
			except Exception as e:
				return_data= {
					'error': '3',
					'message': str(wsd_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			work_start_date_year= work_start_date_year


			#----------------------------Last Account Column--------------------------------------

			try:
				if (data['last_account'].dtype == np.float64 or data['last_account'].dtype == np.int64):
					#app.logger.error(la_err)
					return_data= { 
						'error': '3',
						'message': str(la_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['last_account']= pd.to_datetime(data['last_account'], errors= "ignore")
					last_account_year= data['last_account'].dt.year
					last_account_month= data['last_account'].dt.month
					last_account_day= data['last_account'].dt.day
					last_account_quarter= data['last_account'].dt.quarter
			except Exception as e:
				return_data= {
					'error': '3',
					'message': str(la_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			last_account_year= last_account_year
			last_account_month= last_account_month
			last_account_quarter= last_account_quarter
			last_account_day= last_account_day


			#------------------------------------First Account Column--------------------------------------------


			try:
				if (data['first_account'].dtype == np.float64 or data['first_account'].dtype == np.int64):
					#app.logger.error(fa_err)
					return_data= { 
						'error': '3',
						'message': str(fa_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['first_account']= pd.to_datetime(data['first_account'], errors= "ignore")
					first_account_year= data['first_account'].dt.year
					first_account_month= data['first_account'].dt.month
					first_account_day= data['first_account'].dt.day
					first_account_quarter= data['first_account'].dt.quarter
			except Exception as e:
				return_data= {
					'error': '3',
					'message': str(fa_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			first_account_year= first_account_year
			first_account_month= first_account_month
			first_account_quarter= first_account_quarter
			first_account_day= first_account_day
			


			#--------------------------------------Card Network Column-----------------------------------------------


			try:
				if data['card_network'].values == "Mastercard":
					data['card_network']= le.fit_transform(data['card_network'])
				elif data['card_network'].values == "Visa":
					data['card_network']= le.fit_transform(data['card_network'])
				elif data['card_network'].values == "Verve":
					data['card_network']= le.fit_transform(data['card_network'])
				else:
					#app.logger.error(cn_err)
					return_data= {
						'error': '3',
						'message': str(cn_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(cn_err)
				return_data= {
					'error': '3',
					'message': str(cn_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			card_network= data['card_network']

			#------------------------------------Tier Column ------------------------------------------------------


			try: 
				if data['tier'].values == "Tier 1":
					data['tier']= le.fit_transform(data['tier'])
				elif data['tier'].values == "Tier 2":
					data['tier']= le.fit_transform(data['tier'])
				elif data['tier'].values == "Tier 3":
					data['tier']= le.fit_transform(data['tier'])
				else:
					#app.logger.error(tr_err)
					return_data= {
						'error': '3',
						'message': str(tr_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(tr_err)
				return_data= {
					'error': '3',
					'message': str(tr_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			tier= data['tier']


			#-------------------------------------Marital Status---------------------------------------------------


			try:
				if data['marital_status'].values == "Married":
					data['marital_status']= le.fit_transform(data['marital_status'])
				elif data['marital_status'].values == "Divorced":
					data['marital_status']= le.fit_transform(data['marital_status'])
				elif data['marital_status'].values == "Single":
					data['marital_status']= le.fit_transform(data['marital_status'])
				elif data['marital_status'].values == "Widowed":
					data['marital_status']= le.fit_transform(data['marital_status'])
				else:
					#app.logger.error(ms_err)
					return_data= {
						'error': '3',
						'message': str(ms_err)					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(ms_err)
				return_data= {
					'error': '3',
					'message': str(ms_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			marital_status= data['marital_status']


			#----------------------------------------Selfie_ID_Check Column--------------------------------------------

			try: 
				if data['selfie_id_check'].values == "Successful":
					data['selfie_id_check']= le.fit_transform(data['selfie_id_check'])
				elif data['selfie_id_check'].values == "Failed":
					data['selfie_id_check']= le.fit_transform(data['selfie_id_check'])
				elif data['selfie_id_check'].values == "Pending":
					data['selfie_id_check']= le.fit_transform(data['selfie_id_check'])
				else:
					#app.logger.error(sic_err)
					return_data= {
						'error': '3',
						'message': str(sic_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(sic_err)
				return_data= {
					'error': '3',
					'message': str(sic_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			selfie_id_check= data['selfie_id_check']


			#---------------------------------Work Email Validated ----------------------------------------


			try:
				if data['work_email_validated'].values == 0 or data['work_email_validated'].values == 1:
					data['work_email_validated']= int(data['work_email_validated'])
				else:
					#app.logger.error(wev_err)
					return_data= {
						'error': '3',
						'message': str(wev_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(wev_err)
				return_data= {
					"error": '3',
					"message": str(wev_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			work_email_validated= data['work_email_validated']


			#--------------------------------------------Monthly Net Income Column -----------------------------


			try:
				if data['monthly_net_income'].values == "10,000-54,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "10,000 - 54,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "10,000-55,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "55,000-99,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "55,000-100,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "100,000-199,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "100,000-200,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "200,000-399,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "Above 200,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "400,000-699,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "700,000-999,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "Above 1,000,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "10,000 - 55,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "55,000 - 99,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "55,000 - 100,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "100,000 - 199,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "100,000 - 200,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "200,000 - 399,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "Above200,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "400,000 - 699,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "700,000 - 999,999":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				elif data['monthly_net_income'].values == "Above1,000,000":
					data['monthly_net_income']= le.fit_transform(data['monthly_net_income'])
				
				else:
					#app.logger.error(mni_err)
					return_data= {
						'error': '3',
						'message': str(mni_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			except Exception as e:
				#app.logger.error(mni_err)
				return_data= {
					'error': '3',
					'message': str(e)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			monthly_net_income= data['monthly_net_income']


			#-------------------------------------Bank Column------------------------------------------------------

			try:
				if (data['bank'].dtype == np.float64 or data['bank'].dtype == np.int64):
					#app.logger.error(bk_err)
					return_data= { 
						'error': '3',
						'message': str(bk_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['bank']= le.fit_transform(data['bank'])
			except Exception as e:
				#app.logger.error(bk_err)
				return_data= {
					'error': '3',
					'message': str(bk_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			bank= data['bank']


			#-----------------------------Sector Of Employment----------------------------------------------------

			try:
				if (data['sector_of_employment'].dtype == np.float64 or data['sector_of_employment'].dtype == np.int64):
					#app.logger.error(soe_err)
					return_data= { 
						'error': '3',
						'message': str(soe_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['sector_of_employment']= le.fit_transform(data['sector_of_employment'])
			except Exception as e:
				#app.logger.error(soe_err)
				return_data= {
					'error': '3',
					'message': str(soe_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			sector_of_employment= data['sector_of_employment']


			#-------------------------------------Employment Status Column ------------------------------------

			try:
				if (data['employment_status'].dtype == np.float64 or data['employment_status'].dtype == np.int64):
					#app.logger.error(es_err)
					return_data= { 
						'error': '3',
						'message': str(es_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['employment_status']= le.fit_transform(data['employment_status'])
			except Exception as e:
				#app.logger.error(es_err)
				return_data= {
					'error': '3',
					'message': str(es_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			employment_status= data['employment_status']

			#-----------------------------Purpose Column-----------------------------------------------------

			try:
				if (data['purpose'].dtype == np.float64 or data['purpose'].dtype == np.int64):
					#app.logger.error(pp_err)
					return_data= { 
						'error': '3',
						'message': str(pp_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['purpose']= le.fit_transform(data['purpose'])
			except Exception as e:
				#app.logger.error(pp_err)
				return_data= {
					'error': '3',
					'message': str(pp_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			purpose= data['purpose']

			#------------------------------------Location Column---------------------------------------

			try:
				if (data['location'].dtype == np.float64 or data['location'].dtype == np.int64):
					#app.logger.error(lc_err)
					return_data= { 
						'error': '3',
						'message': str(lc_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['location']= le.fit_transform(data['location'])
			except Exception as e:
				#app.logger.error(lc_err)
				return_data= {
					'error': '3',
					'message': str(lc_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			location= data['location']


			#---------------------------------------Educational Attainment column---------------------

			try:
				if (data['educational_attainment'].dtype == np.float64 or data['educational_attainment'].dtype == np.int64):
					#app.logger.error(ea_err)
					return_data= { 
						'error': '3',
						'message': str(ea_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['educational_attainment']= le.fit_transform(data['educational_attainment'])
			except Exception as e:
				#app.logger.error(ea_err)
				return_data= {
					'error': '3',
					'message': str(ea_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			educational_attainment= data['educational_attainment']


			#--------------------------------No of Dependent Column--------------------------------------------

			try:
				if (data['no_of_dependent'].dtype == np.object):
					#app.logger.error(nod_err)
					return_data= { 
						'error': '3',
						'message': str(nod_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['no_of_dependent']= int(data['no_of_dependent'])
			except Exception as e:
				#app.logger.error(nod_err)
				return_data= {
					'error': '3',
					'message': str(nod_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			no_of_dependent= data['no_of_dependent']


			#----------------------------Phone Network----------------------------------------------

			try:
				if (data['phone_network'].dtype == np.object):
					#app.logger.error(ph_err)
					return_data= { 
						'error': '3',
						'message': str(ph_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					data['phone_network']= le.fit_transform(data['phone_network'])
			except Exception as e:
				#app.logger.error(ph_err)
				return_data= {
					'error': '3',
					'message': str(ph_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')
			phone_network= data['phone_network']
			work_email_validated= int(data['work_email_validated'])

			#-----------------------------------------Card Expiry Column-----------------------------------

			try:
				if (data['card_expiry'].dtype == np.object):
					#app.logger.error(cd_err)
					return_data= { 
						'error': '3',
						'message': str(cd_err)
					}
					return app.response_class(response=json.dumps(return_data), mimetype='application/json')
				else:
					card_expiry_month= data['card_expiry'].map(lambda x: str(int(x))[:-4]).astype(int)
					card_expiry_year= data['card_expiry'].map(lambda x: str(int(x))[-4:]).astype(int)
			except Exception as e:
				#app.logger.error(cd_err)
				return_data= {
					'error': '3',
					'message': str(cd_err)
				}
				return app.response_class(response=json.dumps(return_data), mimetype='application/json')

			#--------------------------------Prediction -------------------------------------------------

			result= [loan_amount, interest_due, tenor, interest_rate, card_network, bank, phone_network, tier, selfie_id_check, marital_status, no_of_dependent, educational_attainment,
			employment_status, sector_of_employment, monthly_net_income, work_email_validated, requested_amount, purpose, proposed_payday, credit_score, location, failed_requests,
			passed_requests, all_requests, logins, phone_numbers, emails, lenders, lending_lenders, loans, card_expiry_month, card_expiry_year, date_of_birth_year, work_start_date_year,
			first_account_year, first_account_month, first_account_day, first_account_quarter, last_account_year, last_account_month, last_account_day, last_account_quarter]

			#Passing data to model & loading model from disk
			#result= result.reshape(-1)
			result= np.array(result)
			classifier= pickle.load(open(model_path, 'rb'))
			prediction= str(classifier.predict([result])[0])
			conf_score= np.max(classifier.predict_proba([result]))*100

			return_data= {
				"error": "0",
				"message": "Successful",
				"prediction": prediction,
				"confidence_score": conf_score.round(2)
			}

		else: 
			return_data= {
				"error": '1',
				"message": "The Parameter body cannot be empty"

			}

	except BadRequest:
		return_data= {
			'error': '2',
			'message': ( 'Non number parameters in JSON format must be in quote' )
		}
	except ValueError as e:
		return_data= {
			'error': '4',
			'message': "Tenor should be an object value in this format ('7 days'), not a number"
		}
		
	return app.response_class(response=json.dumps(return_data), mimetype='application/json')



if __name__ == "__main__":
	app.run(port= 8080, debug= True, host="0.0.0.0")
