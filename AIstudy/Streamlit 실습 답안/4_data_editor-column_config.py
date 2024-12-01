import streamlit as st
import pandas as pd
from datetime import datetime
import pandas as pd

df = pd.DataFrame(
    {
        "name": ["Park", "Lee", "Kim", "Choi"],
        "age": [24, 31, 40, 19],
        "flag": [True, False, True, False],
        "lang": [None, "JavaScript", "Python", "C"],
        "join": [
            datetime(2024, 4, 22, 12, 30),
            datetime(1992, 11, 26, 18, 0),
            datetime(2021, 6, 6, 12, 0),
            datetime(2024, 1, 1, 1, 0),
        ],
        'company_image': ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQnOACrRJyk-4693gXNbbpXfQ4OVXSWm3sl5g&s',
                          'https://i.namu.wiki/i/XSIOAAJo3KQI6q9CtG7Zs35kWdJDGJytIUResnjEOP4v3Z5MpOXUD9KUkg3BrcdqZKNiH73iP9u3bATIX87Fhg.svg',
                          'https://i.namu.wiki/i/Iu9zZQJ6VDhoEC3goMITaHnBNOAbxz5buSqhkMwnNcjBIvbURav08bsIPoY9pn7EBu-Hqlp-x94o6rD8J3vJ2g.svg',
                          'https://i.namu.wiki/i/lku8yP05WBzxHSq95G_lkk_BjXShwWh41RSpSBpbSaqUWD53o_YcbcchVi3ztuVGBCm0fosSMRK3q8xyNzllKg.svg'
        ]
    }
)

## data_editor 안에 column들을 다양한 구성요소들로 채울 수 있다. 확인해보자

st.data_editor(df) ## 날짜를 넣어주면 기본적으로 날짜로 인식하기는 한다. 

## column_config를 사용하면 최소, 최대, 설명, column명 변경 등이 가능하다

st.data_editor(
    df, 
    column_config={
        'join': st.column_config.DatetimeColumn(
            label= 'Join Date', 
            help= '등록일자를 누르시오', 
            min_value= datetime(1992,1,1),
            max_value=datetime(2024,1,20)
        )
    }
)

## Language 같은 경우에 선택 옵션으로 하고 싶다면


st.data_editor(
    df, 
    column_config={
        'join': st.column_config.DatetimeColumn(
            label= 'Join Date', 
            help= '등록일자를 누르시오', 
            min_value= datetime(1992,1,1), 
            max_value=datetime(2024,1,20)),
        'lang': st.column_config.SelectboxColumn(
            label= 'Main Language', 
            help='주 사용 언어를 고르시오', 
            options= [
                'JavaScript',
                'Python',
                'C',
                'Rust'
            ]
        )
    }
)

## 이미지를 직접 그리기

st.data_editor(
    df, 
    column_config={
        'join': st.column_config.DatetimeColumn(
            label= 'Join Date', 
            help= '등록일자를 누르시오', 
            min_value= datetime(1992,1,1), 
            max_value=datetime(2024,1,20)
        ),
        
        'lang': st.column_config.SelectboxColumn(
            label= 'Main Language', 
            help='주 사용 언어를 고르시오', 
            options= [
                'JavaScript',
                'Python',
                'C',
                'Rust'
            ]
        ),
        
        'company_image' : st.column_config.ImageColumn(
            label= 'Company', 
            help = 'company they work'
        )
    }
)
