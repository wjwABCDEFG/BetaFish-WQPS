from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render, redirect
from . import service
from . import data_processor


def index(request):
    return HttpResponse("Hello")


def train(request):
    if request.method == 'GET':
        model_id = request.GET.get("id")
        # model_id=52
        try:
            data = service.train_model(model_id)
            # print("data",data)
            return JsonResponse({'status': 'success', 'data': data})
        except Exception as e:
            return JsonResponse({'status': str(e)})


def predict(request):

    if request.method == 'GET':
        model_id = request.GET.get("id")
        # model_id=52
        print("model_id=", model_id)

        # 时间
        first_month = eval(request.GET.get("startMonth"))
        first_day = eval(request.GET.get("startDay"))
        first_year = eval(request.GET.get("startYear"))
        # 溶解氧
        DO_1 = eval(request.GET.get("DO_1"))
        DO_2 = eval(request.GET.get("DO_2"))
        DO_3 = eval(request.GET.get("DO_3"))
        # 氨氮
        NH3N_1 = eval(request.GET.get("NH3N_1"))
        NH3N_2 = eval(request.GET.get("NH3N_2"))
        NH3N_3 = eval(request.GET.get("NH3N_3"))
        # PH
        PH_1 = eval(request.GET.get("PH_1"))
        PH_2 = eval(request.GET.get("PH_2"))
        PH_3 = eval(request.GET.get("PH_3"))
        # 水温
        WaterTemperature_1 = eval(request.GET.get("waterTemperature_1"))
        WaterTemperature_2 = eval(request.GET.get("waterTemperature_2"))
        WaterTemperature_3 = eval(request.GET.get("waterTemperature_3"))

        # first_month = 1
        # first_day = 2
        # first_year =2018
        # DO_1 = 1.0
        # DO_2 = 2.0
        # DO_3 = 3.0
        # NH3N_1 = 1.0
        # NH3N_2 = 2.0
        # NH3N_3 = 3.0
        # PH_1 = 1.0
        # PH_2 = 2.0
        # PH_3 = 3.0
        # WaterTemperature_1 = 1.0
        # WaterTemperature_2 = 2.0
        # WaterTemperature_3 = 3.0

        first_day, second_day, third_day, Month_days_num = data_processor.get_day(first_year, first_month, first_day)

        data = [first_day, second_day, third_day, DO_1, DO_2, DO_3, NH3N_1, NH3N_2, NH3N_3, PH_1, PH_2,PH_3, WaterTemperature_1, WaterTemperature_2, WaterTemperature_3]
        try:
            data = service.predict_next_month(model_id, [data],Month_days_num)
            print("data",data)
            return JsonResponse({'status': 'success', 'data': data})
        except Exception as e:
            print(e)
            return JsonResponse({'status': str(e)})


def waterquality(request):
    if request.method == 'GET':
        obj = request.GET.get('obj', 'nothing')
        month_num = eval(request.GET.get('month_num', '5'))
        if obj == 'nothing':
            return JsonResponse({'status': 'param error'})
        try:
            data = service.get_last_months_data(obj, month_num)
            return JsonResponse({'status': 'success', 'data': data})
        except Exception as e:
            return JsonResponse({'status': str(e)})
    if request.method == 'POST':
        date = request.POST.get('date', 'nothing')
        ph = eval(request.POST.get('PH', '-1'))
        do = eval(request.POST.get('DO', '-1'))
        nh3n = eval(request.POST.get('NH3N', '-1'))
        if date == 'nothing' or ph == -1 or do == -1 or nh3n == -1:
            return JsonResponse({'status': 'param error'})
        try:
            data = {'date': date, 'PH': ph, 'DO': do, 'NH3N': nh3n}
            service.save_one_waterquality(data)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': str(e)})


def test(request):
    data = service.get_uploaded_waterquality()
    print(data)
    return render(request, 'test.html', {'data': data})


def delete(request):
    id = request.GET.get('id')
    service.delete(id)
    return redirect('/predicter/test/')







