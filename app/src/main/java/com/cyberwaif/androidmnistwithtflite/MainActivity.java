package com.cyberwaif.androidmnistwithtflite;

import android.Manifest;
import android.app.Activity;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.TextView;

import com.cyberwaif.androidmnistwithtflite.R;
import com.cyberwaif.androidmnistwithtflite.view.DrawModel;
import com.cyberwaif.androidmnistwithtflite.view.DrawView;
import com.cyberwaif.utils.PermissionsUtil;

import java.util.Arrays;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends Activity implements View.OnTouchListener {

    private static final String TAG = "MainActivity";

    private static final int PIXEL_WIDTH = 28;

    private TextView mResultText;

    private float mLastX;

    private float mLastY;

    private DrawModel mModel;
    private DrawView mDrawView;

    private View detectButton;

    private PointF mTmpPoint = new PointF();

    private Executor executor = Executors.newSingleThreadExecutor();
    private KerasTFLite mTFLite;

    @SuppressWarnings("SuspiciousNameCombination")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        mDrawView = (DrawView) findViewById(R.id.view_draw);
        mDrawView.setModel(mModel);
        mDrawView.setOnTouchListener(this);

        detectButton = findViewById(R.id.buttonDetect);
        detectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onDetectClicked();
            }
        });

        View clearButton = findViewById(R.id.buttonClear);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClearClicked();
            }
        });

        mResultText = (TextView) findViewById(R.id.textResult);

        PermissionsUtil.checkCriticalPerms(this);
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    if(mTFLite == null)
                        mTFLite = new KerasTFLite(MainActivity.this);
                    makeButtonVisible();
                    Log.d(TAG, "Load Success");
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                detectButton.setVisibility(View.VISIBLE);
            }
        });
    }

    @Override
    protected void onResume() {
        mDrawView.onResume();
        super.onResume();
        if(PermissionsUtil.isPermissionGranted(this, Manifest.permission.READ_EXTERNAL_STORAGE))
            initTensorFlowAndLoadModel();
    }

    @Override
    protected void onPause() {
        mDrawView.onPause();
        super.onPause();
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;

        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;

        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;

        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        mDrawView.calcPos(mLastX, mLastY, mTmpPoint);
        float lastConvX = mTmpPoint.x;
        float lastConvY = mTmpPoint.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        mDrawView.calcPos(x, y, mTmpPoint);
        float newConvX = mTmpPoint.x;
        float newConvY = mTmpPoint.y;
        mModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        mDrawView.invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }

    private void onDetectClicked() {
        float pixels[] = mDrawView.getPixelData();

        //should be same format with train
        for(int i = 0; i < pixels.length; i++){
            pixels[i] = pixels[i]/255;
        }

        for(int i = 0; i<28; i++){
            float[] a = Arrays.copyOfRange(pixels, i*28, i*28+28);
            Log.v(TAG,  Arrays.toString(a));
        }

        String result = mTFLite.run(pixels);

        String value = " Number is : "+result;
        mResultText.setText(value);

    }

    private void onClearClicked() {
        mModel.clear();
        mDrawView.reset();
        mDrawView.invalidate();

        mResultText.setText("");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}

