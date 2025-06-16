# Create your views here.
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from datetime import datetime
import tempfile
import boto3
import torchaudio
import torch
import os
import io
import hmac
import hashlib
import base64
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device
from django.conf import settings
# from .models import Resume
from django.http import JsonResponse

# Create your views here.
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)

audio_path = os.path.join(settings.BASE_DIR, "cloning_sample.wav")
speaker_wav, sampling_rate = torchaudio.load(audio_path)
speaker = model.make_speaker_embedding(speaker_wav, sampling_rate)

# S3 업로드
s3_client = boto3.client('s3')
bucket_name = settings.AWS_TTS_BUCKET_NAME


@api_view(['POST'])
@permission_classes([IsAuthenticated])
# @permission_classes([AllowAny])  # 인증 없이 Postman에서 테스트 가능
def generate_followup_question(request):
    text = request.data.get('text')
    question_number = request.data.get('question_number')
    user = request.user

    print("type(text):", type(text))
    print("text raw:", repr(text))
    if not text:
        return Response({'error': 'text field is required'}, status=400)

    try:
        # 텍스트와 스피커 임베딩으로 conditioning 구성
        cond_dict = make_cond_dict(
            text=text,
            speaker=speaker,
            language="ko",
            emotion=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            speaking_rate=23.0,
            pitch_std=20.0,
        )
        conditioning = model.prepare_conditioning(cond_dict)

        # Zonos 모델로 음성 생성
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes).cpu()

         # 메모리 버퍼 생성
        buffer = io.BytesIO()
        # 메모리 버퍼에 wav 저장
        torchaudio.save(buffer, wavs[0], model.autoencoder.sampling_rate, format="wav")
        buffer.seek(0)  # 버퍼 위치 초기화

        today = datetime.now().strftime("%m%d")
        email_prefix = user.email.split('@')[0]
        filename = f"질문 {question_number}.wav"
        s3_key = f'tts_outputs/{email_prefix}/{today}/{filename}'  # 원하면 고유 이름으로 변경
        s3_client.upload_fileobj(buffer, bucket_name, s3_key)

        file_url = f'https://{bucket_name}.s3.amazonaws.com/{s3_key}'

        response = {
            "message": "TTS 생성 및 S3 업로드 성공",
            "file_url": file_url
        }

        return Response(response, status=200)

    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def generate_resume_question(request):
    bucket = settings.AWS_QUESTION_BUCKET_NAME
    user_email = request.user.email.split('@')[0]
    prefix = f"{user_email}/"

    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        files = sorted([obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.txt')])
        if not files:
            return Response({"error": "No text files found in your S3 folder."}, status=404)

        generated_files = []

        for key in files:
            # 텍스트 읽기
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            s3.download_fileobj(Bucket=bucket, Key=key, Fileobj=temp)
            temp.close()
            with open(temp.name, 'r', encoding='utf-8') as f:
                text = f.read().strip()

            cond_dict = make_cond_dict(
                text=text,
                speaker=speaker,
                language="ko",
                emotion=[0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                speaking_rate=23.0,
                pitch_std=20.0,
            )
            conditioning = model.prepare_conditioning(cond_dict)
            codes = model.generate(conditioning)
            wavs = model.autoencoder.decode(codes).cpu()

            buffer = io.BytesIO()
            torchaudio.save(buffer, wavs[0], model.autoencoder.sampling_rate, format="wav")
            buffer.seek(0)

            today = datetime.now().strftime("%m%d")
            filename = f"{os.path.basename(key).replace('.txt', '')}.wav"
            s3_key = f'tts_outputs/{user_email}/{today}/{filename}'
            s3_client.upload_fileobj(buffer, bucket_name, s3_key)

            file_url = f'https://{bucket_name}.s3.amazonaws.com/{s3_key}'
            generated_files.append({"text_file": key, "tts_file_url": file_url})

        return Response({
            "message": "TTS 생성 및 S3 업로드 성공 (batch)",
            "results": generated_files
        }, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
