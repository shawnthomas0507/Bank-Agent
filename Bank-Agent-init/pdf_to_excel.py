import pandas as pd

def excel_to_pdf(file_path):
    df = pd.read_excel(file_path,header=1)
    df = df.dropna(subset=df.columns[1:], how='all')
    df.loc[0:2, 'Balance'] = df.loc[0:2, 'Deposit']
    df.loc[2:23, 'Balance'] = df.loc[2:23, 'Category']

    df.loc[0:1, 'Deposit'] = None
    df=df.drop(['Category'],axis=1)
    df=df.fillna(0)
    return df


def save_to_csv():
    df=excel_to_pdf("C:\\Users\\shawn\\.credentials\\pnc document (1).xlsx")
    df.to_csv('output.csv', index=False)




