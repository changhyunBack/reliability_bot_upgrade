import streamlit as st
import numpy as np
import json
import scipy.stats as stats
import pandas as pd 
import matplotlib.pyplot as plt
from reliability.ALT_fitters import Fit_Lognormal_Exponential,Fit_Weibull_Exponential,Fit_Exponential_Exponential,Fit_Normal_Exponential,Fit_Everything_ALT
from lifelines import WeibullAFTFitter,LogNormalAFTFitter
from reliability.Fitters import Fit_Exponential_1P,Fit_Lognormal_2P,Fit_Weibull_2P,Fit_Normal_2P
from scipy.stats import norm
from autograd.differential_operators import hessian
import os
from scipy.stats import chisquare   
import warnings
from scipy.stats import expon,norm
import json
warnings.filterwarnings('ignore')
from langchain.tools import tool

# 그래프 경로설정 꼬이지 않게 잘하기
current_dir = os.path.dirname(os.path.abspath(__file__))

def save_plot(fig, filename):
    # 폴더가 없으면 생성
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    fig.savefig(filepath)
    return filepath

# 고급 통계 분석
def calculate_normal_confidence_intervals(mu, sigma, mu_var, sigma_var, cov, alpha, time, CI_type="two_sided"):
    z = (time-mu)/sigma
    varz = (mu_var + z**2 *sigma_var+2*z*cov)/sigma**2

    if CI_type == "two_sided":
        z1 = z - norm.ppf(1 - alpha/2) * np.sqrt(varz)
        z2 = z + norm.ppf(1 - alpha/2) * np.sqrt(varz)

        result=pd.DataFrame({
            "time": time,
            "CDF": norm.cdf(z),
            "lower": norm.cdf(z1),
            "upper": norm.cdf(z2),
        })
    elif CI_type == "lower":
        zL = z - norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "lower": psev(zL,0,1),
        }
    elif CI_type == "upper":
        zU = z + norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "upper": psev(zU,0,1),}
    return result

def calculate_lognormal_confidence_intervals(mu, sigma, mu_var, sigma_var, cov, alpha, time, CI_type="two_sided"):
    z = (np.log(time) - mu) / sigma
    varz = (mu_var + z**2 *sigma_var+2*z*cov)/sigma**2
    
    if CI_type == "two_sided":
        z1 = z - norm.ppf(1 - alpha/2) * np.sqrt(varz)
        z2 = z + norm.ppf(1 - alpha/2) * np.sqrt(varz)
        
        result=pd.DataFrame({
            "time": time,
            "CDF": norm.cdf(z),
            "lower": norm.cdf(z1),
            "upper": norm.cdf(z2),
        })
    elif CI_type == "lower":
        zL = z - norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "lower": psev(zL,0,1),
        }
    elif CI_type == "upper":
        zU = z + norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "upper": psev(zU,0,1),
        }
    return result

def pexp2(q,mean):  # cdf
    return 1-np.exp(-q/mean)
def qexp2(p,mean):
    return -np.log(1-p)*mean
def psev(q,mu,sigma):
    return 1-np.exp(-np.exp((q-mu)/sigma))

def calculate_exponential_confidence_intervals2(lam,target, mean_upper_CI,mean_lower_CI):
    mean_estimate= 1/lam
    percentiles_from_values = expon.cdf(target, scale=mean_estimate) * 100

    # Confidence intervals for the percentiles from values
    lower_percentiles_CI = expon.cdf(target, scale=mean_lower_CI) * 100
    upper_percentiles_CI = expon.cdf(target, scale=mean_upper_CI) * 100

    # Create a DataFrame to display the resultsㅁ
    results = pd.DataFrame({
        'time': target,
        'percentile': percentiles_from_values,
        'lower': lower_percentiles_CI,
        'upper': upper_percentiles_CI
    })

    return results

def calculate_weibull_confidence_intervals(scale, shape, scale_var, shape_var, cov, alpha, time, CI_type="two_sided"):
    z = (np.log(time) - np.log(scale)) / (1/shape)
    varz = z**2 * shape_var / shape**2 + shape**2 * scale_var / scale**2 - 2 * z * cov / scale

    if CI_type == "two_sided":
        z1 = z - norm.ppf(1 - alpha/2) * np.sqrt(varz)
        z2 = z + norm.ppf(1 - alpha/2) * np.sqrt(varz)
        result=pd.DataFrame({
            "time": time,
            "CDF": (psev(z,0,1)),
            "lower": (psev(z1,0,1)),
            "upper": (psev(z2,0,1)),
        })
    elif CI_type == "lower":
        zL = z - norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "lower": psev(zL,0,1),
        }
    elif CI_type == "upper":
        zU = z + norm.ppf(1 - alpha) * np.sqrt(varz)
        return {
            "time": time,
            "CDF": psev(z,0,1),
            "upper": psev(zU,0,1),
        }
    return result

# 백분위수와 cdf신뢰구간을 모두 계산한다.
def calculate_oters(name,result,percent,target,failures,alpha,right_censored):
    percent,target= np.array(percent),np.array(target)
    if name == "normal":
        params = [result.mu,result.sigma]
        hessian_matrix = hessian(Fit_Normal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),)
        covariance_matrix = np.linalg.inv(hessian_matrix)
        tmp_var = [covariance_matrix[0,0],covariance_matrix[1,1]]
        total_cov = covariance_matrix[0][1]
        
        zp=norm.ppf(percent)
        xp=params[0]+zp*params[1]
        mu_var = tmp_var[0]
        sigma_var = tmp_var[1]
        var_xp=mu_var+zp**2*sigma_var+2*zp*total_cov
        se=np.sqrt(var_xp)
        lower=xp-norm.ppf(1-alpha/2)*np.sqrt(var_xp)
        upper=xp+norm.ppf(1-alpha/2)*np.sqrt(var_xp)
        
        result1=pd.DataFrame({'percent':percent, 'percenttile':xp, 'SE': se, 'lower':lower,'upper':upper})
        result2=pd.DataFrame(calculate_normal_confidence_intervals(params[0],params[1],tmp_var[0],tmp_var[1],total_cov,alpha,target)).set_index('time')
    
    elif name=='lognormal':
        params = [result.mu,result.sigma]
        hessian_matrix = hessian(Fit_Lognormal_2P.LL)(
                np.array(tuple(params)),
                np.array(tuple(failures)),
                np.array(tuple(right_censored)),)
        covariance_matrix = np.linalg.inv(hessian_matrix)
        tmp_var = [covariance_matrix[0,0],covariance_matrix[1,1]]
        total_cov = covariance_matrix[0][1]
        
        zp=norm.ppf(percent)
        xp=np.exp(params[0]+zp*params[1])
        var_xp=xp**2*(tmp_var[0]+zp**2*tmp_var[1]+2*zp*total_cov)
        se=np.sqrt(var_xp)
        lower=np.exp(np.log(xp)-norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
        upper=np.exp(np.log(xp)+norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
        result1=pd.DataFrame({'percent':percent, 'percenttile':xp, 'SE': se, 'lower':lower,'upper':upper})
        result2=pd.DataFrame(calculate_lognormal_confidence_intervals(params[0],params[1],tmp_var[0],tmp_var[1],total_cov,alpha,target)).set_index('time')

    elif name=='exponential':
        params = result.Lambda # 람다 추정치
        mean_estimate = 1 / params # 평균 추정치
        se = result.Lambda_SE
        lam_lower = result.Lambda_lower
        lam_upper = result.Lambda_upper
        mean_lower_CI = 1 / lam_upper  
        mean_upper_CI = 1 / lam_lower
        percentile_values = expon.ppf(np.array(percent),scale=mean_estimate)
        percentile_std_errors = se / params ** 2 * expon.ppf(np.array(percent))
        # Confidence intervals for the percentiles
        lower = expon.ppf(np.array(percent), scale=mean_lower_CI)
        upper = expon.ppf(np.array(percent), scale=mean_upper_CI)
        result1 = pd.DataFrame({'percent':percent,"percentile":percentile_values,'SE':percentile_std_errors,'lower':lower,'upper':upper})
        result2 = pd.DataFrame(calculate_exponential_confidence_intervals2(lam=params,target=target,mean_lower_CI=mean_lower_CI,mean_upper_CI=mean_upper_CI)).set_index('time')
        
    else: # 와이블
        params = [result.alpha, result.beta]
        hessian_matrix = hessian(Fit_Weibull_2P.LL)(
            np.array(tuple(params)),
            np.array(tuple(failures)),
            np.array(tuple(right_censored)),
        )
        covariance_matrix = np.linalg.inv(hessian_matrix)
        tmp_var = [covariance_matrix[0][0],covariance_matrix[1][1]]
        total_cov =abs(covariance_matrix[0][1])
        
        zp = np.log(-np.log(1-percent))
        xp = params[0]*(-np.log(1-percent))**(1/params[1])
        var_xp = (xp/params[0])**2*tmp_var[0]+xp**2/params[1]**4*zp**2*tmp_var[1]-2*(zp*xp**2)/(params[0]*params[1]**2)*abs(total_cov)
        se = np.sqrt(var_xp)

        lower = np.exp(np.log(xp)-norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
        upper = np.exp(np.log(xp)+norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
        result1=pd.DataFrame({'percent':percent, 'percenttile':xp, 'SE': se, 'lower':lower,'upper':upper})
        result2=pd.DataFrame(calculate_weibull_confidence_intervals(params[0],params[1],tmp_var[0],tmp_var[1],total_cov,alpha,target)).set_index('time')
    return result1,result2

# func1 (개별 스트레t스대해서 모든 분포에 대한 추정)
@tool
def find_individual_dist(life_span,stress,cens=None,target=[1000],percent=[0.01,0.05,0.1],alpha=0.05):
    """
    This function estimates the distribution of data at each individual stress level.\
    require value:\
    -life_span(column name)\
    -stress(column name).\
    Optional value:\
    -cens(Column name that displays 관측중단 data)\
    -alpha(Significance level)\
    -percent(Life at a certain cumulative failure rate (B10 represents a cumulative failure rate of 10%)\
    -target(Cumulative failure rate over a certain lifetime)\
    """    
    # 데이터 확인 
    if st.session_state['selected_filename'] == 'None':
        return '데이터업로드 필요'
    
    df_origin=st.session_state['dataframes'][st.session_state['selected_filename']]
    df=df_origin.copy()
    
    # 열 존재여부 확인
    for col in [life_span,stress,cens]:
        if col and col not in df.columns:
            return f'{col}은 잘못된 열이름'
    # (스트레스1,분포1), (스트레스1,분포2)... 이런순서
    figs=[]
    # 스트레스별로 모수값인데, idx 0=정규,지수,로그노말,와이블
    results={}
    fit=[]

    fitters = [Fit_Normal_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Weibull_2P]
    names =['normal','exponential','lognormal','weibull']
    
    # 스트레스별 
    unique_stress= df[stress].unique()
    for us in unique_stress:
        results[us]={}
        data= df.loc[df[stress]==us]
        
        if cens:
            # 정상관측
            failures = data.loc[data[cens]==1,life_span].values 
            # 우측관측중단 
            right_censored = data.loc[data[cens]==0,life_span].values
        else:
            failures= df[life_span].values
            right_censored=[]
            
        # 분포추정
        for i in range(len(fitters)):
            results[us][names[i]]={}
            fig, ax = plt.subplots(1, 1)
            result = fitters[i](failures=failures,right_censored=right_censored, print_results=False,show_probability_plot=True)
            results[us][names[i]]['results']=result.results
            results[us][names[i]]['fit']=result.goodness_of_fit
            fit.append(result.results)
            figs.append(fig)
            plt.close()
            
            r1,r2=calculate_oters(name=names[i],result=result,percent=percent,target=target,failures=failures,right_censored=right_censored,alpha=alpha)
            
            results[us][names[i]]['r1']=r1
            results[us][names[i]]['r2']=r2
   
    # 결과표시 
    for i in range(len(unique_stress)):
        for j in range(len(names)):
            st.session_state['messages'].append({'role':'assistant','content': f"<h3>{unique_stress[i]},{names[j]}</h3>"})
            st.session_state['messages'].append({'role':'df_result','content':results[unique_stress[i]][names[j]]['results']})
            st.session_state['messages'].append({'role':'df_result','content':results[unique_stress[i]][names[j]]['fit']})
            st.session_state['messages'].append({'role':'df_result','content':results[unique_stress[i]][names[j]]['r1']})
            st.session_state['messages'].append({'role':'df_result','content':results[unique_stress[i]][names[j]]['r2']})
            filename = f"plot_{len([msg for msg in st.session_state['messages'] if msg['role'] == 'plot']) + 1}.png"
            filepath = save_plot(figs[4*i+j], filename)
            st.session_state['messages'].append({'role':'plot', 'content':filepath})
    st.rerun()
    
# func2(최적분포와 모델을 찾아주는 함수)
@tool
def find_best_dist(life_span,stress,cens=None):
    """
    This is a function that tells you the optimal distribution, which distribution the data best follows: normal, lognormal, Weibull, or exponential.\
    require value:\
    -life_span(column name)\
    -stress(column name).\
    Optional value:\
    -cens(Column name that displays 관측중단 data)\
    """
    # 데이터 확인 
    if st.session_state['selected_filename'] == 'None':
        return '데이터업로드 필요'
    
    df_origin=st.session_state['dataframes'][st.session_state['selected_filename']]
    df=df_origin.copy()

    # 열 존재여부 확인
    for col in [life_span,stress,cens]:
        if col and col not in df.columns:
            return f'{col}은 잘못된 열이름'
     
    # 관측중단이 있는경우
    if cens:
        failures= df.loc[df[cens]==1,life_span].values
        failures_stress= df.loc[df[cens]==1,stress].values
        right_censored = df.loc[df[cens]==0,life_span].values
        right_censored_stress = df.loc[df[cens]==0,stress].values
        # 분포분석
        result=Fit_Everything_ALT(failures=failures,failure_stress=failures_stress,right_censored=right_censored,right_censored_stress=right_censored_stress,show_probability_plot=True, print_results=False,show_life_stress_plot=False,show_best_distribution_probability_plot=True)
        # 그래프
        probability_plot = result.probability_plot
        best_distribution_plot= result.best_distribution_probability_plot

    # 관측중단이 없는경우
    else:
        failures= df[life_span].values
        failures_stress= df[stress].values
        # 분포분석
        result=Fit_Everything_ALT(failures=failures,failure_stress=failures_stress,show_probability_plot=True, print_results=False,show_life_stress_plot=False,show_best_distribution_probability_plot=True)
        # 그래프
        probability_plot = result.probability_plot
        best_distribution_plot= result.best_distribution_probability_plot
       
    
    plt.close('all')
    r= result.results.iloc[:, [0] + list(range(7, 11))]
    result_info = {'result' : r.loc[r['ALT_model']==result.best_model_name].to_dict()}
    # 데이터프레임 저장
    st.session_state['messages'].append({'role':'df_result','content':r})
    # 그래프 저장 
    # 1
    filename1 = f"plot_{len([msg for msg in st.session_state['messages'] if msg['role'] == 'plot']) + 1}.png"
    filepath1 = save_plot(probability_plot, filename1)
    st.session_state['messages'].append({'role':'plot', 'content':filepath1})
    # 2
    filename2 = f"plot_{len([msg for msg in st.session_state['messages'] if msg['role'] == 'plot']) + 1}.png"
    filepath2 = save_plot(best_distribution_plot, filename2)
    st.session_state['messages'].append({'role':'plot', 'content':filepath2})
   
    # 화면에 보여주기 
    with st.session_state['placeholder_plot'].container():
        st.dataframe(r)
        
        st.pyplot(probability_plot)
        st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>{filename1}</p>", unsafe_allow_html=True)
    
        st.pyplot(best_distribution_plot)
        st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>{filename2}</p>", unsafe_allow_html=True)
    
    # 모델에게 돌려줄값 
    return json.dumps(result_info) 

def Lognormal_Arrenius(df,life_span,stress,cens,use_stress,target,alpha,percent):
    k= 1/11604.83
    # 관측중단이 있는경우
    if cens:
        # 우측관측중단 
        right_censored=df.loc[df[cens]==0,life_span].values
        right_censored_stress =df.loc[df[cens]==0,stress].values
        # 정상관측
        failures= df.loc[df[cens]==1,life_span].values
        failures_stress= df.loc[df[cens]==1,stress].values

        result=Fit_Lognormal_Exponential(failures=failures,failure_stress=failures_stress,
                                    right_censored=right_censored,
                        right_censored_stress=right_censored_stress,show_probability_plot=True, print_results=False,
                        show_life_stress_plot=True,use_level_stress=use_stress,CI=1-alpha)
        
        # 그래프 담아두기
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')

        result.distribution_at_use_stress.CDF()        
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
            
        df['acc'] = 11604.83/(df[stress])
        
        aft =LogNormalAFTFitter(alpha=alpha)
        aft.fit(df,duration_col=life_span,event_col=cens, formula="acc", show_progress=False)
    
    # 정상관측만 있는경우 
    else:
        failures= df[life_span].values
        failures_stress= df[stress].values
        
        result=Fit_Lognormal_Exponential(failures=failures,failure_stress=failures_stress,
                                   show_probability_plot=True, print_results=False,show_life_stress_plot=True,use_level_stress=use_stress,CI=1-alpha)
        
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')

        result.distribution_at_use_stress.CDF()
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
        
        df['acc'] = 11604.83/(df[stress])
        
        aft =LogNormalAFTFitter(alpha=alpha)
        aft.fit(df,duration_col=life_span, formula="acc", show_progress=False)

    # 가속수명분석을 통한 회귀식 추출
    coef = aft.params_.values[0:2]
    scale = aft.params_[-1]

    # 추정한 회귀모델 계수를 활용해 편도함수 계산
    a=np.array([0,0,np.exp(scale)])
    b=np.array([1, 11604.83 / (use_stress),0])
    tmp_delta = pd.DataFrame({'b':b,'a':a})

    # delta method를 활용해 회귀식의 분산-공분산 행렬 계산
    tmp_cov = tmp_delta.T.values @ aft.variance_matrix_.values @tmp_delta.values
    tmp_var = [tmp_cov[0,0],tmp_cov[1,1]]
    # 모수간 공분산
    total_cov = tmp_cov[0,1]

    # 사용조건하에서의 모수 계산
    values = np.array([1, 11604.83 / (use_stress)])
    distpar = [coef@[1,11604.83/(use_stress)],np.exp(scale)] # 척도, 형상모수 계산
    distparse = np.sqrt(np.diag(tmp_cov))# 척도, 형상모수 분산계산 

    mu=distpar[0]
    sigma=distpar[1]
    zp=norm.ppf(percent)
    xp=np.exp(mu+zp*sigma)
    var_xp=xp**2*(tmp_var[0]+zp**2*tmp_var[1]+2*zp*total_cov)
    se=np.sqrt(var_xp)
    lower=np.exp(np.log(xp)-norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
    upper=np.exp(np.log(xp)+norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)

    # 특정 cdf에서 백분위수(수명)의 점추정치와 신뢰구간
    result1=pd.DataFrame({'percent':percent, 'percenttile':xp, 'SE': se, 'lower':lower,'upper':upper})
    
    # 아레니우스 회귀분석표
    # 계수
    COEF = [aft.params_.values[0],aft.params_.values[1],np.exp(aft.params_).values[2]]
    # 표준오차
    SE=[aft.standard_errors_.values[0],aft.standard_errors_.values[1],result.sigma_SE]
    # Z
    Z=[aft.summary['z'].values[0],aft.summary['z'].values[1],None]
    # P
    P=aft.summary['p'].values[0],aft.summary['p'].values[1],None
    # 하한
    LOWER=aft.confidence_intervals_.values[0][0],aft.confidence_intervals_.values[1][0],np.exp(aft.confidence_intervals_.values[2,0])
    # 상한
    UPPER=aft.confidence_intervals_.values[0][1],aft.confidence_intervals_.values[1][1],np.exp(aft.confidence_intervals_.values[2,1])
    
    regr=pd.DataFrame({'coef':COEF,'SE':SE,'Z':Z,'P':P,'lower':LOWER,'upper':UPPER},index=['intercept','temp','shape'])

    # 필수값 정리 
    loglik = result.loglik
    stress = result.change_of_parameters
       
    result2=calculate_lognormal_confidence_intervals(mu=mu,sigma=sigma,mu_var=tmp_var[0],sigma_var=tmp_var[1],cov=total_cov,alpha=alpha,time=target)
    return [loglik,alpha,result.mean_life],stress,result1,regr,figures,figure_cdf,result2
    
def Weibull_Arrenius(df,life_span,stress,cens,use_stress,target,alpha,percent):
    # 관측중단이 있는 경우
    if cens:
        # 우측관측중단
        right_censored=df.loc[df[cens]==0,life_span].values
        right_censored_stress =df.loc[df[cens]==0,stress].values
        # 정상관측
        failures= df.loc[df[cens]==1,life_span].values
        failures_stress= df.loc[df[cens]==1,stress].values
        
        result=Fit_Weibull_Exponential(failures=failures,failure_stress=failures_stress,
                                    right_censored=right_censored,
                        right_censored_stress=right_censored_stress,show_probability_plot=True, print_results=False,
                        show_life_stress_plot=True,use_level_stress=use_stress,CI=1-alpha)
        
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
             
        result.distribution_at_use_stress.CDF()
        
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
      
        df['acc'] = 11604.83/(df[stress])
        
        aft = WeibullAFTFitter(alpha=alpha)
        aft.fit(df,duration_col=life_span,event_col=cens, formula="acc", show_progress=False)
    
    # 정상관측만 있는경우
    else:
        failures= df[life_span].values
        failures_stress= df[stress].values
        
        result=Fit_Weibull_Exponential(failures=failures,failure_stress=failures_stress,
                        show_probability_plot=True, print_results=False,
                        show_life_stress_plot=True,use_level_stress=use_stress)
        
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
        
        result.distribution_at_use_stress.CDF()
        
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all') 
        
        df['acc'] = 11604.83/(df[stress])
        
        aft = WeibullAFTFitter(alpha=alpha)
        aft.fit(df,duration_col=life_span, formula="acc", show_progress=False)

    # 가속수명분석을 통한 회귀식 추출
    coef = aft.params_.values[0:2]
    scale = aft.params_[-1]

    # 추정한 회귀모델 계수를 활용해 편도함수 계산
    a=np.array([0,0,-np.exp(scale)])
    b=np.exp(coef[0:2].dot(np.array([1, 11604.83 / (use_stress)])))*np.array([1,11604.83/(use_stress),0])
    tmp_delta = pd.DataFrame({'b':b,'a':a})

    # delta method를 활용해 회귀식의 분산-공분산 행렬 계산
    tmp_cov = tmp_delta.T.values @ aft.variance_matrix_.values @tmp_delta.values
    tmp_var = [tmp_cov[0,0],tmp_cov[1,1]]
    # 모수간 공분산
    total_cov = np.abs(tmp_cov[0,1])

    # 사용조건하에서의 모수 계산
    distpar = [np.exp(coef@[1,11604.83/(use_stress)]),np.exp(scale)] # 척도, 형상모수 계산
    distparse = np.sqrt(np.diag(tmp_cov))# 척도, 형상모수 분산계산
    
    tg_scale = distpar[0]
    tg_shape = distpar[1]

    # log(-log(1-percent)) -> weibull 분포 백분위수 계산 공식
    zp = np.log(-np.log(1-percent))

    # weibull 분포 역함수를 활용해 백분위수 계산
    xp = tg_scale*(-np.log(1-percent))**(1/tg_shape)

    # delta method를 활용하여 백분위수의 분산을 근사적으로 계산
    var_xp = (xp/tg_scale)**2*tmp_var[0]+xp**2/tg_shape**4*zp**2*tmp_var[1]-2*(zp*xp**2)/(tg_scale*tg_shape**2)*abs(total_cov)
    se = np.sqrt(var_xp)

    # weibull 분포의 백분위수는 로그 정규 분포르 따르기 때문에 로그변환을 적용하여 정규분포 기반 신뢰구간을 계산한 후 이를 다시 변환
    lower = np.exp(np.log(xp)-norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
    upper = np.exp(np.log(xp)+norm.ppf(1-alpha/2)*np.sqrt(var_xp)/xp)
    
    # 특정 cdf에서 백분위수(수명)의 점추정치와 신뢰구간
    result1=pd.DataFrame({'percent':percent, 'percenttile':xp, 'SE': se, 'lower':lower,'upper':upper})
    
    # 아레니우스 회귀분석표
    # 계수
    COEF = [aft.params_.values[0],aft.params_.values[1],np.exp(aft.params_).values[2]]
    # 표준오차
    SE=[aft.standard_errors_.values[0],aft.standard_errors_.values[1],result.beta_SE]
    # Z
    Z=[aft.summary['z'].values[0],aft.summary['z'].values[1],None]
    # P
    P=aft.summary['p'].values[0],aft.summary['p'].values[1],None
    # 하한
    LOWER=aft.confidence_intervals_.values[0][0],aft.confidence_intervals_.values[1][0],np.exp(aft.confidence_intervals_.values[2,0])
    # 상한
    UPPER=aft.confidence_intervals_.values[0][1],aft.confidence_intervals_.values[1][1],np.exp(aft.confidence_intervals_.values[2,1])
    
    regr=pd.DataFrame({'coef':COEF,'SE':SE,'Z':Z,'P':P,'lower':LOWER,'upper':UPPER},index=['intercept','temp','shape'])
    
    # 필수값 정리 
    loglik = result.loglik
    stress = result.change_of_parameters
      
    result2=calculate_weibull_confidence_intervals(tg_scale,tg_shape,tmp_var[0],tmp_var[1],total_cov,alpha,time=target)
    return [loglik,alpha,result.mean_life],stress,result1,regr,figures,figure_cdf,result2

# 이부분 제거 할거임. 확인후에
def calculate_exponential_confidence_intervals(scale,se,alpha, time, CI_type="two_sided"):
    distpar = scale
    results=[]
    for t in time:
        probability =pexp2(t, mean=distpar)
        stderr=np.sqrt((-np.log(1-probability))**2 * se **2)
        quantile = qexp2(probability,mean=distpar)
        if CI_type == "two_sided":
            zval=stats.norm.ppf(1-alpha/2,0,1)
            lower = pexp2(quantile/np.exp(zval*stderr/quantile),mean=distpar)
            upper = pexp2(quantile*np.exp(zval*stderr/quantile),mean=distpar)
            results.append({
                "time": t,
                "CDF": probability,
                "lower": lower,
                "upper": upper,
            })
    return pd.DataFrame(results)
        
def Exponential_Arrenius(df,life_span,stress,cens,use_stress,target,alpha,percent):
    k= 1/11604.83
    # 관측중단이 있는경우
    if cens:
        # 우측관측중단 
        right_censored=df.loc[df[cens]==0,life_span].values
        right_censored_stress =df.loc[df[cens]==0,stress].values
        # 정상관측
        failures= df.loc[df[cens]==1,life_span].values
        failures_stress= df.loc[df[cens]==1,stress].values
        
        result=Fit_Exponential_Exponential(failures=failures,failure_stress=failures_stress,
                                    right_censored=right_censored,
                        right_censored_stress=right_censored_stress,show_probability_plot=True, print_results=False,
                        show_life_stress_plot=True,use_level_stress=use_stress,CI=1-alpha)
        
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')

        result.distribution_at_use_stress.CDF()
        
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')
                
    # 정상관측만 있는경우 
    else:
        failures= df[life_span].values
        failures_stress= df[stress].values
        
        result=Fit_Exponential_Exponential(failures=failures,failure_stress=failures_stress,
                                   show_probability_plot=True, print_results=False,show_life_stress_plot=True,use_level_stress=use_stress,CI=1-alpha)
        
        figures= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')

        result.distribution_at_use_stress.CDF()
        
        figure_cdf= [plt.figure(i) for i in plt.get_fignums()]  
        plt.close('all')


    def LL(params, t_f, t_rc, T_f, T_rc):  # log likelihood function
        # failure times
        LL_f = Fit_Exponential_Exponential.logf(t_f, T_f, params[0], params[1]).sum()
        # right censored times
        LL_rc = Fit_Exponential_Exponential.logR(t_rc, T_rc, params[0], params[1]).sum()
        return -(LL_f + LL_rc)

    zval=stats.norm.ppf(1-alpha/2)
    params=[result.a,result.b]  
    hessian_matrix = hessian(LL)(
        np.array(tuple(params)),
        np.array(tuple(failures)),
        np.array(tuple(right_censored)),
        np.array(tuple(failures_stress)),
        np.array(tuple(right_censored_stress)),
    )
    # a,b에 대한 공분산행렬
    covariance_matrix = np.linalg.inv(hessian_matrix)
    
    # 역변환해서 intercept,temp의 공분산행렬을 찾기
    # temp
    s_aa=(k**2)*covariance_matrix[0][0] 
    # 절편
    s_bb=(1/result.b)**2 * covariance_matrix[1][1]
    # 공분산
    s_ab = k * (1/result.b) * covariance_matrix[0][1]
    # 공분산행렬
    variance_matrix = np.array([[s_bb,s_ab],[s_ab,s_aa]])
    
    # 델타메소드를 사용해서 신뢰구간 계산
    # 가속수명분석을 통한 회귀식 추출
    coef=np.array([np.log(result.b),result.a*k])
    # 추정한 회귀모델 계수를 활용해 편도함수 계산
    tmp_delta=np.exp(coef.dot(np.array([1, 11604.83 / use_stress])))*np.array([1,11604.83/use_stress])
    # # delta method를 활용해 회귀식의 분산-공분산 행렬 계산
    tmp_cov = tmp_delta.T @ variance_matrix @tmp_delta
    # 사용조건하에서의 모수 계산
    distpar = np.exp(coef@[1,11604.83/use_stress]) 
    distparse = np.sqrt(tmp_cov)

    theta= distpar
    xp=-np.log(1-percent)*theta
    var_xp=(-np.log(1-percent))**2*distparse**2
    lower = np.exp(np.log(xp)-zval*np.sqrt(var_xp)/xp)
    upper = np.exp(np.log(xp)+zval*np.sqrt(var_xp)/xp)
    
    # 특정 cdf에서 백분위수(수명)의 점추정치와 신뢰구간
    result1=pd.DataFrame({'percent':percent,'percentile':xp,'se':np.sqrt(var_xp),'lower':lower,'upper':upper})
    
    # 회귀분석표 
    COEF=np.array([np.log(result.b),result.a*k])
    SE =np.array([result.b_SE/result.b,result.a_SE*k])
    Z = COEF/SE
    if stats.norm.cdf(Z[0])>0.5:
        p1=(1-stats.norm.cdf(Z[0]))*2
    else:
        p1=stats.norm.cdf(Z[0])*2
    if stats.norm.cdf(Z[1])>0.5:
        p2=(1-stats.norm.cdf(Z[1]))*2
    else:
        p2=stats.norm.cdf(Z[1])*2
    P=[p1,p2]
    lower ,upper= COEF - zval*SE,COEF + zval*SE
    regr=pd.DataFrame({'coef':COEF,'SE':SE,'Z':Z,'P':P,'lower':lower,'upper':upper},index=['intercept','temp'])

    # 필수값 정리 
    loglik = result.loglik
    stress = result.change_of_parameters

    # 특정수명에대한 누적고장률 
    result2=calculate_exponential_confidence_intervals(scale=distpar,se=distparse,alpha=alpha,time=target)
    return [loglik,alpha,result.mean_life],stress,result1,regr,figures,figure_cdf,result2

# func3(전체 분석함수)
@tool
def analyze_AFT(life_span,stress,use_stress,cens=None,target=[1000],percent=[0.01,0.05,0.1],alpha=0.05,fix_dist='weibull'):    
    """
    This function analyzes accelerated life tests.\
    require value:\
    -life_span(column name)\
    -stress(column name).\
    -use_stress : Temperature or stress of actual use environment\
    Optional value:\
    -cens(Column name that displays 관측중단 data)\
    -alpha(Significance level)\
    -percent(Life at a certain cumulative failure rate (B10 represents a cumulative failure rate of 10%)\
    -target(Cumulative failure rate over a certain lifetime)\
    -fix_dist : A factor that fixes the distribution to be used during analysis. 와이블,정규,로그정규,지수라고 말해도 이 인자는 weibull,normal,lognormal and exponential로 사용한다.
    """
    # 데이터 확인 
    if st.session_state['selected_filename'] == 'None':
        return '데이터업로드 필요'
       
    percent= np.array(percent)
    target= np.array(target)

    df_origin=st.session_state['dataframes'][st.session_state['selected_filename']]
    df=df_origin.copy()
    
    # 열 존재여부 확인
    for col in [life_span,stress,cens]:
        if col and col not in df.columns:
            return f'{col}은 잘못된 열이름'
    idx=None
    if fix_dist =='normal':
        idx=0
    elif fix_dist=='exponential':
        idx=1
    elif fix_dist=='lognormal':
        idx=2
    elif fix_dist=='weibull':
        idx=3
    # 분포를 사용해서 분석
    analyzers =[None,Exponential_Arrenius, Lognormal_Arrenius, Weibull_Arrenius]
    analyzer= analyzers[idx]

    R=analyzer(df,life_span=life_span,stress=stress,cens=cens,use_stress=use_stress,percent=percent,target=target,alpha=alpha)
    
    # 동일모수검정(가속성) 지수분포는 동일형상모수를 항상만족(shape=1)
    param = ['original sigma', 'common shape', 'original sigma','original beta']
    observed_shapes = R[1][param[idx]].values
    expected_shape = sum(observed_shapes) / len(observed_shapes)
    chi2_statistic, chi2_p_value = chisquare(f_obs=observed_shapes, f_exp=[expected_shape]*len(observed_shapes))
    
    R0=R[0]
    R0=pd.DataFrame({'loglik':[R0[0]],'alpha':[R0[1]],'mean of use_stress':[R0[2]]})
    # 동일성검정값 추가
    R = R + (pd.DataFrame([{'pvalue':chi2_p_value}]),)
    
    ## 결과값 저장
    st.session_state['messages'].append({'role':'df_result','content':R0})
    # 데이터프레임
    l1= [R[1],R[2],R[3],R[6].set_index('time'),R[7]]
    l2= ['추정된 모수의 값','백분위수 수명','회귀계수표','누적고장률','동일성검정']
    for i in range(len(l1)):
        st.session_state['messages'].append({'role':'header','content':l2[i]})
        st.session_state['messages'].append({'role':'df_result','content':l1[i]})
    
    # 그래프
    l3= [R[4][0],R[4][1],R[5][0]]
    l4= ['시간에 따른 누적확률','평균수명','누적고장률']
    filenames = [] # 일시적으로 표시할 캡션값
    for i in range(len(l3)):
        filename = f"plot_{len([msg for msg in st.session_state['messages'] if msg['role'] == 'plot']) + 1}.png"
        filenames.append(filename)
        filepath = save_plot(l3[i], filename)
        st.session_state['messages'].append({'role':'assistant','content': f"<h3>{l4[i]}</h3>"})
        st.session_state['messages'].append({"role": "plot", "content": filepath})  
    
    with st.session_state['placeholder_plot'].container():
        st.subheader('로그 가능성')
        st.dataframe(R0)
        
        st.subheader('추정된 모수의 값')
        st.dataframe(R[1])
        
        st.subheader('백분위수 수명')  
        st.dataframe(R[2])
        
        st.subheader('회귀계수표')
        st.dataframe(R[3])
        
        st.subheader('누적고장률')
        st.dataframe(R[6].set_index('time'))
        
        st.subheader('동일성검정')
        st.dataframe(R[7])
        
        st.subheader('확률지')
        st.pyplot(R[4][0])
        st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>{filenames[0]}</p>", unsafe_allow_html=True)
       
        st.subheader('평균수명')
        st.pyplot(R[4][1])
        st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>{filenames[1]}</p>", unsafe_allow_html=True)
       
        st.subheader('시간에 따른 누적확률')
        st.pyplot(R[5][0])
        st.markdown(f"<p style='text-align: center; font-size:14px; color:gray;'>{filenames[2]}</p>", unsafe_allow_html=True)
        
    result_info={'loglik':R0.to_dict(),'a':R[1].to_dict(),'b':R[2].to_dict(),'c':R[3].to_dict(),'d':R[6].to_dict(),'Equal parameter test':R[7].to_dict()}
    return json.dumps(result_info)


# 합불판정을 위한 무고장 시험 테스트
@tool
def calculate_lifetime_or_test_time(param, dist, confidence_level, target_lifetime, target_reliability,test_time=None, sample_size=None):
    """
    Calculate the number of samples or test time required for fault-free reliability testing.\
    require value:\
    -param : Parameter values ​​of the distribution. Weibull is the shape, the rest is scale\
    -dist : Lifespan distribution\
    -target_lifetime\ 
    -target_reliability\
    Optional value:\
    -sample_size or test_time : Enter only one of the two
    """
    # 와이블 분포의 역신뢰도 함수 R^(-1)(p)
    def weibull_inverse_reliability(p, theta, beta):
        return theta * (-np.log(p)) ** (1 / beta)
    # 와이블 분포의 신뢰도 함수 R(t)
    def weibull_reliability(t, theta, beta):
        return np.exp(- (t / theta) ** beta)
    # 정규 분포의 신뢰도 함수 R(t)
    def normal_reliability(t, mu, sigma):
        return 1 - norm.cdf((t - mu) / sigma)
    # 정규 분포의 역신뢰도 함수 R^(-1)(p)
    def normal_inverse_reliability(p, mu, sigma):
        return mu + sigma * norm.ppf(1 - p)
    # 지수 분포의 역신뢰도 함수 R^(-1)(p)
    def exponential_inverse_reliability(p, theta, beta):
        return theta * (-np.log(p)) ** (1 / beta)
    # 지수 분포의 신뢰도 함수 R(t)
    def exponential_reliability(t, theta, beta):
        return np.exp(- (t / theta) ** beta)
    # 로그 정규 분포의 역신뢰도 함수 R^(-1)(p)
    def lognormal_inverse_reliability(p, mu, sigma):
        return np.exp(sigma * norm.ppf(1 - p) + mu)
    # 로그 정규 분포의 신뢰도 함수 R(t)
    def lognormal_reliability(t, mu, sigma):
        return 1 - norm.cdf((np.log(t) - mu) / sigma)
    
    if sample_size is None and test_time is None:
        raise ValueError("At least one of 'target_lifetime' or 'test_time' must be provided.")
    elif sample_size is not None and test_time is not None:
        raise ValueError("only input one")
  
    alpha = 1 - confidence_level
    percent = 1- target_reliability
    # 시험 시간 계산: 표본수를 알고있는경우
    if sample_size is not None: # param : beta
      if dist=='weibull':
        beta=param
        theta = target_lifetime / (-np.log(target_reliability)) ** (1 / beta)
        # 필요한 시간 계산
        required_time = weibull_inverse_reliability(alpha ** (1 / sample_size), theta, beta)
        return required_time
        
      elif dist=='exponential':
        beta=1
        theta = target_lifetime / (-np.log(target_reliability)) ** (1 / param)
        required_time = weibull_inverse_reliability(alpha ** (1 / sample_size), theta, beta)
        return 0
        
      elif dist=='normal':
        # 목표 신뢰도에 따른 필요한 시험 시간 계산
        required_time = normal_inverse_reliability(alpha ** (1 / sample_size), mu, sigma)
        return 0
        
      elif dist=='lognormal':
        sigma= param
        mu = np.log(target_lifetime) - norm.ppf(percent) * 0.5 
        required_time = lognormal_inverse_reliability(alpha ** (1 / sample_size), mu, sigma)
      else:
        raise ValueError("only use normal,lognormal,weibull,exponential")
      
      result = {'Blife' : percent,'required_time' : required_time} 
      return json.dumps(result)

    # 표본수 계산 : 시험 시간을 알고 있는 경우    
    elif test_time is not None:
      if dist=='weibull':
        beta=param
        theta = target_lifetime / (-np.log(target_reliability)) ** (1 / beta)
        # 주어진 시간에서 고장나지 않을 확률 계산
        no_failure_probability = weibull_reliability(test_time, theta, beta)
        # 필요한 표본 수 계산
        required_sample_size = np.log(alpha) / np.log(no_failure_probability)
   
      elif dist=='exponential':
        beta=1
        theta = target_lifetime / (-np.log(target_reliability)) ** (1 / param)
        no_failure_probability = weibull_reliability(test_time, theta, beta)
        required_sample_size = np.log(alpha) / np.log(no_failure_probability)
              
      elif dist=='normal':
        sigma = param
        mu = target_lifetime
        no_failure_probability = normal_reliability(test_time, mu, sigma)
        required_sample_size = np.log(alpha) / np.log(no_failure_probability)

      elif dist=='lognormal':
        sigma= param
        mu = np.log(target_lifetime) - norm.ppf(percent) * 0.5 
        no_failure_probability = lognormal_reliability(test_time, mu, sigma)
        required_sample_size = np.log(alpha) / np.log(no_failure_probability)
      else:
        raise ValueError("only use normal,lognormal,weibull,exponential")
      
      result = {'Blife' : percent,'required_sample_size' : required_sample_size} 
      return json.dumps(result)
