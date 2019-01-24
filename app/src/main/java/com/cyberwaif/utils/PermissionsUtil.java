package com.cyberwaif.utils;

import android.Manifest;
import android.annotation.TargetApi;
import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PermissionsUtil {

    public static final int REQUEST_EXTERNAL_STORAGE = 2;
    public static final String PERMS_READ_EXTERNAL_STORAGE = Manifest.permission.READ_EXTERNAL_STORAGE;
    public static final String PERMS_WRITE_EXTERNAL_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    public static final int DURATION_CRITICAL = 300;
    public static boolean DEBUG = false;
    private static String TAG = "PermissionsUtil";
    private static RequestingPerms mRequestingPerms;

    public static boolean isPermissionGranted(Context context, String permission) {
        int status = 0;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            status = context.checkSelfPermission(permission);
        }
        if (DEBUG) {
            Log.i(TAG, "Permission " + permission + ", checkSelfPermission " + status);
        }
        return (status == PackageManager.PERMISSION_GRANTED);
    }

    public static void requestPermissions(Activity activity, int requestCode, String... permissions) {
        Log.e(TAG,"permission request "+ Arrays.toString(permissions)+"-------------------------");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            activity.requestPermissions(permissions, requestCode);
        }
        if (mRequestingPerms == null) {
            mRequestingPerms = new RequestingPerms(requestCode, permissions);
        } else {
            mRequestingPerms.clear();
            mRequestingPerms.set(requestCode, permissions);
        }
    }

    public static boolean checkCriticalPerms(final Activity activity) {
        ArrayList<String> perms = new ArrayList<>();
        int code = 0;

        if (!isPermissionGranted(activity.getApplicationContext(), PERMS_READ_EXTERNAL_STORAGE) ||
                !isPermissionGranted(activity.getApplicationContext(), PERMS_WRITE_EXTERNAL_STORAGE)) {
            perms.add(PERMS_READ_EXTERNAL_STORAGE);
            perms.add(PERMS_WRITE_EXTERNAL_STORAGE);
            code += REQUEST_EXTERNAL_STORAGE;
        }

        if (code == 0) return true;

        final int requestCode = code;
        final String[] request = perms.toArray(new String[perms.size()]);

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                requestPermissions(activity, requestCode, request);
            }
        }, DURATION_CRITICAL);
        return false;
    }

    public static class RequestingPerms {
        public int code;
        public String[] perms;

        public RequestingPerms(int requestCode, String... permissions) {
            set(requestCode, permissions);
        }

        public void set(int requestCode, String... permissions) {
            code = requestCode;
            perms = permissions;
        }

        public void clear() {
            code = 0;
            perms = null;
        }
    }
}
