#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import base64  # Standard Python Module
import numpy as np
from io import StringIO, BytesIO  # Standard Python Module

@import url(http://fonts.googleapis.com/earlyaccess/nanumgothic.css);



#plt.rc('font', family='NanumBarunGothic')
#plt.rc('axes', unicode_minus=False)

# with open( "style.css" ) as css:
#     st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)







def generate_excel_download_link(df):
    # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
    towrite = BytesIO()
    df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download= "data_download.xlsx">Download Excel File</a>'
    return st.markdown(href, unsafe_allow_html=True)




st.set_page_config(page_title='Excel로 올려서 컷을 다운받자')
st.title('전형별 OX산포도 만들기? 📈')
st.subheader('Feed me with your Excel file')

uploaded_file = st.file_uploader('XLSX 형식의 파일을 올려주세요', type='xlsx')
if uploaded_file:
    st.markdown('전형명, 모집단위(코드포함), 등록여부, 산출등급')
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.dataframe(df)
    
    df.columns=['전형유형','학과명','코드','합격차수','평균등급']
    
    
    choice = df['전형유형'].unique()
    
    
    choice_column = st.selectbox('선택해주세요',choice, )
    
    data11 = df[df['전형유형'] == choice_column]
    
    shushu = len(data11['학과명'].unique())
                
    
    합격0 = data11[data11['합격차수'].notnull()]
    불합격0 = data11[data11['합격차수'].isnull()]
    su = [x+0.1 for x in range(shushu)]
    gp = 합격0.groupby(['코드'])['평균등급'].mean()
    ggg = pd.DataFrame(gp.round(2))
    shen  = ggg['평균등급'].tolist()
    items = 합격0['학과명'].unique().tolist()

    qq1  = 합격0.groupby(['학과명']).mean()
                
    
    import seaborn as sns
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    #def PlotEach():
    
    ax = sns.stripplot(x=불합격0['평균등급'], y=불합격0['학과명'], data=불합격0, marker="x",s=14,color='#111111' ,jitter=False, alpha=1, linewidth=1)


    pnts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pnts) / 3, -np.cos(pnts) / 3]
    vert = np.r_[circ, circ[::-1] * .7]

    open_circle = mpl.path.Path(vert)

    ax = sns.stripplot(x=합격0['평균등급'], y=합격0['학과명'], data=합격0, marker=open_circle,s=17, color='#5954ED', jitter = False)





    ax.figure.set_size_inches(15, 23)
    ax.set_xlim([1,9])
    ax.set_xticks((1,2,3,4,5,6,7,8,9))
    ax.set_xticklabels(['1등급','2등급','3등급','4등급','5등급','6등급','7등급','8등급','9등급'])    


    ax.set_title('2023학년도 대진대학교 OX 산포도'+'\n', fontsize=25)

    ax.tick_params(right=False, top=True, labelright=False, labeltop=True)  # 모두 True일 경우 x축 2개 y축 2개

    ax.tick_params (axis = 'x', labelsize =12)
    ax.tick_params (axis = 'y', labelsize =15)
    ax.set_xlabel('', fontsize=10, color = '#5D5D5D')
    ax.set_ylabel('', fontsize=10, color = '#5D5D5D')
    #ax.legend(['O 합격(충원합격포함)' +'\n'+'X 불합격' ],fontsize =11, loc =(0.85,1.03))



    plt.text(7.60,-0.05,'O' ,fontsize=17, color='#5954ED')
    plt.text(7.62,0.34,'X' ,fontsize=16, color='#202020')
    plt.text(7.78,-0.1,'합격(충원합격포함)' ,fontsize=15, color='black')
    plt.text(7.78,0.32,'불합격' ,fontsize=15, color='black')
    plt.gca().add_patch(Rectangle((7.55,-0.4),1.4,0.85,linewidth=1,edgecolor='#E4E4E4',facecolor='#E4E4E4')) 

    for i in su:
        plt.text(1.1,i,'평균: '+ str(shen[int(i-0.1)]) ,fontsize=13, color='#006699') #평균을 집어넣자
        plt.gca().add_patch(Rectangle((1.08,-0.2+int(i-0.1)),0.6,0.58,linewidth=1,edgecolor='#006699',facecolor='none'))  # 네모칸을 쳐보자


    plt.gca().add_patch(Rectangle((1.08,3.8),0.6,0.58,linewidth=1,edgecolor='#006699',facecolor='none'))    


    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle=':', color='gray')
    ax.set_facecolor('w')


    ax.spines["bottom"].set_color("#5E5E5E")
    ax.spines["left"].set_color("#5E5E5E")
    ax.spines["top"].set_color("#5E5E5E")
    ax.spines["right"].set_color("#5E5E5E")

    plt.gcf().subplots_adjust(left=0.2)

    png = plt.savefig('대진대ox산포도.png')
    
            
    st.set_option('deprecation.showPyplotGlobalUse', False)            
    st.pyplot(png)
    
    
#     st.download_button(
#     label="Download Pic as Png",
#     data=png,
#     file_name='대진대ox산포도.png',
#     mime='image/png',
# )
    
     
    

    # -- GROUP DATAFRAME
#     output_columns = ['Sales', 'Profit']
#     df_grouped = df.groupby(by=[groupby_column], as_index=False)[output_columns].sum()


    # -- DOWNLOAD SECTION
#   st.subheader('Downloads:')
#   generate_excel_download_link(dfc1)
#   generate_html_download_link(m)






