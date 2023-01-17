#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st  # pip install streamlit
import pandas as pd  # pip install pandas
import base64  # Standard Python Module
import numpy as np
from io import StringIO, BytesIO  # Standard Python Module
import matplotlib.pyplot as plt
import koreanize_matplotlib






def generate_excel_download_link(df):
    # Credit Excel: https://discuss.streamlit.io/t/how-to-add-a-download-excel-csv-function-to-a-button/4474/5
    towrite = BytesIO()
    df.to_excel(towrite, encoding="utf-8", index=False, header=True)  # write to BytesIO buffer
    towrite.seek(0)  # reset pointer
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download= "data_download.xlsx">Download Excel File</a>'
    return st.markdown(href, unsafe_allow_html=True)






#st.set_page_config(page_title='Excel upload')
st.title('전형별 OX산포도 만들기? 📈')
st.subheader('Feed me with your Excel file')

st.text_input('대학명 입력', '대진대학교')
color = st.color_picker('OX에서 O의 색을 결정해주세요', '#5954ED')

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
	
	
    #matplotlib.rcParams['font.family']='NanumBarunGothic'
    #matplotlib.rcParams['axes.unicode_minus'] =False
    #def PlotEach():
    # plt.rc('font', family='NanumBarunGothic')
	
	
    # sns.set(font="NanumBarunGothic", rc={"axes.unicode_minus":False}, style='darkgrid')


    pnts = np.linspace(0, np.pi * 2, 24)
    circ = np.c_[np.sin(pnts) / 3, -np.cos(pnts) / 3]
    vert = np.r_[circ, circ[::-1] * .7]

    open_circle = mpl.path.Path(vert)

    ax = sns.stripplot(x=합격0['평균등급'], y=합격0['학과명'], data=합격0, marker=open_circle,s=17, color=color, jitter = False)


    

    ax = sns.stripplot(x=불합격0['평균등급'], y=불합격0['학과명'], data=불합격0, marker="x",s=14,color='#111111' ,jitter=False, alpha=1, linewidth=1)



	
	





    ax.figure.set_size_inches(15, 23)
    ax.set_xlim([1,9])
    ax.set_xticks((1,2,3,4,5,6,7,8,9))
    ax.set_xticklabels(['1등급','2등급','3등급','4등급','5등급','6등급','7등급','8등급','9등급'])    


    ax.set_title('2023학년도 '+ daxue + choice_column +' OX 산포도'+'\n', fontsize=25)

    ax.tick_params(right=False, top=True, labelright=False, labeltop=True)  # 모두 True일 경우 x축 2개 y축 2개

    ax.tick_params (axis = 'x', labelsize =12)
    ax.tick_params (axis = 'y', labelsize =15)
    ax.set_xlabel('', fontsize=10, color = '#5D5D5D')
    ax.set_ylabel('', fontsize=10, color = '#5D5D5D')
    #ax.legend(['O 합격(충원합격포함)' +'\n'+'X 불합격' ],fontsize =11, loc =(0.85,1.03))



    plt.text(7.60,-0.05,'O' ,fontsize=17, color='#5954ED')
    plt.text(7.62,0.34,'X' ,fontsize=16, color='#202020')
    plt.text(7.78,-0.1,'합격(충원합격포함)' ,fontsize=13, color='black')
    plt.text(7.78,0.32,'불합격' ,fontsize=13, color='black')
    plt.gca().add_patch(Rectangle((7.55,-0.4),1.4,0.85,linewidth=1,edgecolor='#E4E4E4',facecolor='#E4E4E4')) 

    for i in su:
        plt.text(1.1,i,'평균: '+ str(shen[int(i-0.1)]) ,fontsize=11, color='#006699') #평균을 집어넣자
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

    png = '대진대ox산포도.png'
    png1 = plt.savefig(png)
    
            
    st.set_option('deprecation.showPyplotGlobalUse', False)            
    st.pyplot(png1)
	
    


    st.subheader('boxplot')
	
    ax= sns.boxplot(x='학과명',y='평균등급',data=합격0)

    plt.xticks(rotation = - 90 )
    plt.rcParams['figure.figsize'] = (20, 10)
    ax.figure.set_size_inches(25, 15)


    ax.set_ylim([1,9])
    ax.set_yticks((1,2,3,4,5,6,7,8,9))
    ax.set_yticklabels(['1등급','2등급','3등급','4등급','5등급','6등급','7등급','8등급','9등급'])


    ax.tick_params(right=True, top=False, labelright=True, labeltop=False)  # 모두 True일 경우 x축 2개 y축 2개

    ax = plt.gca()
    ax.grid(which='major', axis='both', linestyle=':', color='gray')
    ax.set_facecolor('w')


    plt.gca().invert_yaxis()
    plt.xlabel('모집단위명', color = 'k', fontsize = 25)
    plt.ylabel('과목평균등급', color = 'k', fontsize = 25)
    plt.rcParams.update({'font.size': 20})
    plt.gcf().subplots_adjust(bottom=0.37)
	
    box1 = '대진대박스플롯.png' 	
    box2 = plt.savefig(box1)
    st.pyplot(box2)
	
	
    # Save to file first or an image file has already existed.
   
	

	


    
    
    
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






