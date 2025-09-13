#!/bin/bash
cd android
./gradlew assembleDebug
echo "APK built: app/build/outputs/apk/debug/app-debug.apk"