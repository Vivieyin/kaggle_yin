
from sys import prefix
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='white',context='notebook',palette='muted')
import matplotlib.pyplot as plt

train=pd.read_csv('c:/Users/xin/Desktop/yzw/pro/train.csv')
test=pd.read_csv('c:/Users/xin/Desktop/yzw/pro/test.csv')

# print('实验数据大小:',train.shape)
# print('预测数据大小:',test.shape)
full=train.append(test,ignore_index=True)
full.describe()

# full.info()

sns.barplot(data=train,x='Embarked',y='Survived')
# print('Embarked为"S"的乘客，其生存率为%.2f'%full['Survived'][full['Embarked']=='S'].value_counts(normalize=True)[1])

sns.factorplot('Pclass',col='Embarked',data=train,kind='count',size=3)
sns.barplot(data=train,x='Parch',y='Survived')
sns.barplot(data=train,x='SibSp',y='Survived')
sns.barplot(data=train,x='Pclass',y='Survived')
sns.barplot(data=train,x='Sex',y='Survived')

#创建坐标轴
ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
#作图，选择图形类型
ageFacet.map(sns.kdeplot,'Age',shade=True)
#其他信息：坐标轴范围、标签等
ageFacet.set(xlim=(0,train['Age'].max()))
ageFacet.add_legend()

ageFacet=sns.FacetGrid(train,hue='Survived',aspect=3)
ageFacet.map(sns.kdeplot,'Fare',shade=True)
ageFacet.set(xlim=(0,150))
ageFacet.add_legend()

farePlot=sns.distplot(full['Fare'][full['Fare'].notnull()],label='skewness:%.2f'%(full['Fare'].skew()))
farePlot.legend(loc='best')
full['Fare']=full['Fare'].map(lambda x: np.log(x) if x>0 else 0)

#对Cabin缺失值进行处理，利用U（Unknown）填充缺失值
full['Cabin']=full['Cabin'].fillna('U')
full['Cabin'].head()

#对Embarked缺失值进行处理，查看缺失值情况
full[full['Embarked'].isnull()]
full['Embarked'].value_counts()
full['Embarked']=full['Embarked'].fillna('S')

full[full['Fare'].isnull()]
full['Fare']=full['Fare'].fillna(full[(full['Pclass']==3)&(full['Embarked']=='S')&(full['Cabin']=='U')]['Fare'].mean())

#构造新特征Title
full['Title']=full['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
#查看title数据分布
full['Title'].value_counts()

#将title信息进行整合
TitleDict={}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss'
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Mrs'
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

full['Title']=full['Title'].map(TitleDict)
full['Title'].value_counts()

#可视化分析Title与Survived之间关系
sns.barplot(data=full,x='Title',y='Survived')

full['familyNum']=full['Parch']+full['SibSp']+1
#查看familyNum与Survived
sns.barplot(data=full,x='familyNum',y='Survived')

def familysize(familyNum):
    if familyNum==1:
        return 0
    elif (familyNum>=2)&(familyNum<=4):
        return 1
    else:
        return 2

full['familySize']=full['familyNum'].map(familysize)
full['familySize'].value_counts()

#查看familySize与Survived
sns.barplot(data=full,x='familySize',y='Survived')

#提取Cabin字段首字母
full['Deck']=full['Cabin'].map(lambda x:x[0])
#查看不同Deck类型乘客的生存率
sns.barplot(data=full,x='Deck',y='Survived')

#提取各票号的乘客数量
TickCountDict={}
TickCountDict=full['Ticket'].value_counts()
TickCountDict.head()

#将同票号乘客数量数据并入数据集中
full['TickCot']=full['Ticket'].map(TickCountDict)
full['TickCot'].head()

#查看TickCot与Survived之间关系
sns.barplot(data=full,x='TickCot',y='Survived')

#按照TickCot大小，将TickGroup分为三类。
def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2
#得到各位乘客TickGroup的类别
full['TickGroup']=full['TickCot'].map(TickCountGroup)
#查看TickGroup与Survived之间关系
sns.barplot(data=full,x='TickGroup',y='Survived')


#查看缺失值情况
full[full['Age'].isnull()].head()

#筛选数据集
AgePre=full[['Age','Parch','Pclass','SibSp','Title','familyNum','TickCot']]
#进行one-hot编码
AgePre=pd.get_dummies(AgePre)
ParAge=pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge=pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge=pd.get_dummies(AgePre['Pclass'],prefix='Pclass')
#查看变量间相关性
AgeCorrDf=pd.DataFrame()
AgeCorrDf=AgePre.corr()
AgeCorrDf['Age'].sort_values()

#拼接数据
AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)
AgePre.head()

#拆分实验集和预测集
AgeKnown=AgePre[AgePre['Age'].notnull()]
AgeUnKnown=AgePre[AgePre['Age'].isnull()]

#生成实验数据的特征和标签
AgeKnown_X=AgeKnown.drop(['Age'],axis=1)
AgeKnown_y=AgeKnown['Age']

AgeUnKnown_X=AgeUnKnown.drop(['Age'],axis=1)

#利用随机森林构建模型
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(random_state=None,n_estimators=500,n_jobs=-1)
rfr.fit(AgeKnown_X,AgeKnown_y)
# 模型得分
rfr.score(AgeKnown_X,AgeKnown_y)

AgeUnKnown_y=rfr.predict(AgeUnKnown_X)
full.loc[full['Age'].isnull(),['Age']]=AgeUnKnown_y

full.info()

#提取乘客的姓氏及相应的乘客数
full['Surname']=full['Name'].map(lambda x:x.split(',')[0].strip())
SurnameDict={}
full['SurnameNum']=full['Surname'].map(SurnameDict)

#将数据分为两组
MaleDf=full[(full['Sex']=='male')&(full['Age']>12)&(full['familyNum']>=2)]
FemChildDf=full[((full['Sex']=='female')|(full['Age']<=12))&(full['familyNum']>=2)]


MSurNamDf=MaleDf['Survived'].groupby(MaleDf['Surname']).mean()
MSurNamDf.head()
MSurNamDf.value_counts()


MSurNamDict={}
MSurNamDict=MSurNamDf[MSurNamDf.values==1].index


FCSurnamDf=FemChildDf['Survived'].groupby(FemChildDf['Surname']).mean()
FCSurnamDf.head()
FCSurnamDf.value_counts()

FCSurnamDict={}
FCSurnamDict=FCSurnamDf[FCSurnamDf.values==0].index
FCSurnamDict

full.loc[(full['Survived'].isnull())&(full['Sex']=='male')&(full['Surname'].isin(MSurNamDict)),'Age']=5
full.loc[(full['Survived'].isnull())&(full['Sex']=='male')&(full['Surname'].isin(MSurNamDict)),'Sex']='female'

full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurnamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Age']=60
full.loc[(full['Survived'].isnull())&(full['Surname'].isin(FCSurnamDict))&((full['Sex']=='female')|(full['Age']<=12)),'Sex']='male'

fullSel=full.drop(['Cabin','Name','Ticket','PassengerId','Surname','SurnameNum'],axis=1)
corrDf=pd.DataFrame()
corrDf=fullSel.corr()
corrDf['Survived'].sort_values(ascending=True)

plt.figure(figsize=(8,8))
sns.heatmap(fullSel[['Survived','Age','Embarked','Fare','Parch','Pclass','Sex','SibSp','Title','familyNum','familySize','Deck','TickCot','TickGroup']].corr(),cmap='BrBG',annot=True,linewidth=5)

plt.xticks(rotation=45)

fullSel=fullSel.drop(['familyNum','Parch','SibSp','TickCot'],axis=1)
fullSel=pd.get_dummies(fullSel)
PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
TickgroupDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')
FamilySizeDf=pd.get_dummies(full['TickGroup'],prefix='TickGroup')


fullSel=pd.concat(['fullSel','PclassDf','TickgroupDf','FamilySizeDf'],axis=1)

experData=fullSel[fullSel'Survived'.notnull()]
preData=fullSel[fullSel'Survived'.isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1 )

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

kfold=StratifiedKFold(n_splits=10)

classifliers=[]
classifliers.append(SVC())



