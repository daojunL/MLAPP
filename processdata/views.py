from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


def home(request):
    """the homepage view"""
    return render(request, 'template.html')

@csrf_exempt
@require_POST
def input(request):
    name = request.POST.get('name', None)
    file = request.POST.get('file', None)
    print(name)
    print(type(file))
    content = {
        'name' : name
    }
    return  render(request, 'processdata.html', context={'content': content})
