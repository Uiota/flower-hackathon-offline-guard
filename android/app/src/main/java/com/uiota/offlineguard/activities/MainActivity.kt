package com.uiota.offlineguard.activities

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.uiota.offlineguard.R
import com.uiota.offlineguard.services.NetworkMonitorService
import com.uiota.offlineguard.services.OfflineGuardService
import com.uiota.offlineguard.utils.GuardianManager
import com.uiota.offlineguard.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var networkMonitor: NetworkMonitorService
    private lateinit var offlineGuard: OfflineGuardService
    private lateinit var guardianManager: GuardianManager
    
    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 100
        private const val STORAGE_PERMISSION_REQUEST = 101
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        initializeServices()
        setupUI()
        requestPermissions()
    }
    
    private fun initializeServices() {
        networkMonitor = NetworkMonitorService(this)
        offlineGuard = OfflineGuardService(this)
        guardianManager = GuardianManager(this)
        
        // Set up network monitoring callback
        networkMonitor.setNetworkCallback { isOnline ->
            runOnUiThread {
                updateNetworkStatus(isOnline)
                if (!isOnline) {
                    activateOfflineMode()
                } else {
                    deactivateOfflineMode()
                }
            }
        }
    }
    
    private fun setupUI() {
        binding.apply {
            // Initialize Guardian character
            val guardian = guardianManager.getCurrentGuardian()
            guardianNameText.text = guardian.name
            guardianClassText.text = guardian.guardianClass
            guardianLevelText.text = "Level ${guardian.level}"
            
            // Set up button listeners
            generateQrButton.setOnClickListener {
                generateOfflineProof()
            }
            
            scanQrButton.setOnClickListener {
                scanQRCode()
            }
            
            safeModeButton.setOnClickListener {
                toggleSafeMode()
            }
            
            // Update initial status
            updateNetworkStatus(networkMonitor.isNetworkAvailable())
        }
    }
    
    private fun requestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.ACCESS_NETWORK_STATE
        )
        
        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        
        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsToRequest.toTypedArray(),
                CAMERA_PERMISSION_REQUEST
            )
        }
    }
    
    private fun updateNetworkStatus(isOnline: Boolean) {
        binding.apply {
            networkStatusText.text = if (isOnline) "Online" else "Offline"
            networkStatusIndicator.setBackgroundColor(
                if (isOnline) 
                    ContextCompat.getColor(this@MainActivity, android.R.color.holo_green_light)
                else 
                    ContextCompat.getColor(this@MainActivity, android.R.color.holo_red_light)
            )
            
            // Update Guardian status based on offline time
            if (!isOnline) {
                val offlineDuration = offlineGuard.getOfflineDuration()
                guardianStatusText.text = "Offline Guard Active - ${formatDuration(offlineDuration)}"
            } else {
                guardianStatusText.text = "Guardian Monitoring"
            }
        }
    }
    
    private fun activateOfflineMode() {
        offlineGuard.startOfflineMode()
        
        // Show offline activation notification
        Toast.makeText(this, "🛡️ Offline Guard Activated!", Toast.LENGTH_SHORT).show()
        
        // Update UI for offline mode
        binding.apply {
            safeModeIndicator.setBackgroundColor(
                ContextCompat.getColor(this@MainActivity, android.R.color.holo_orange_light)
            )
            generateQrButton.isEnabled = true
        }
        
        // Evolve Guardian character
        guardianManager.recordOfflineEvent()
    }
    
    private fun deactivateOfflineMode() {
        val offlineDuration = offlineGuard.stopOfflineMode()
        
        if (offlineDuration > 0) {
            Toast.makeText(this, "Back online! Offline duration: ${formatDuration(offlineDuration)}", Toast.LENGTH_LONG).show()
            
            // Reward Guardian for offline time
            guardianManager.rewardOfflineTime(offlineDuration)
        }
        
        // Update UI for online mode
        binding.apply {
            safeModeIndicator.setBackgroundColor(
                ContextCompat.getColor(this@MainActivity, android.R.color.darker_gray)
            )
        }
    }
    
    private fun generateOfflineProof() {
        if (!networkMonitor.isNetworkAvailable()) {
            val proof = offlineGuard.generateOfflineProof()
            if (proof != null) {
                // Show QR code
                showQRCode(proof.qrCodeData)
                Toast.makeText(this, "✅ Offline proof generated!", Toast.LENGTH_SHORT).show()
                
                // Update Guardian stats
                guardianManager.recordProofGeneration()
            } else {
                Toast.makeText(this, "❌ Failed to generate proof", Toast.LENGTH_SHORT).show()
            }
        } else {
            Toast.makeText(this, "Device must be offline to generate proof", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun scanQRCode() {
        // Launch QR scanner - simplified for demo
        Toast.makeText(this, "📷 QR Scanner launching...", Toast.LENGTH_SHORT).show()
        // In full implementation, would use CameraX or ZXing
    }
    
    private fun toggleSafeMode() {
        val isSafeModeActive = offlineGuard.toggleSafeMode()
        
        binding.apply {
            safeModeButton.text = if (isSafeModeActive) "Disable Safe Mode" else "Enable Safe Mode"
            safeModeIndicator.setBackgroundColor(
                if (isSafeModeActive)
                    ContextCompat.getColor(this@MainActivity, android.R.color.holo_red_light)
                else
                    ContextCompat.getColor(this@MainActivity, android.R.color.darker_gray)
            )
        }
        
        val message = if (isSafeModeActive) "🔒 Safe Mode Activated" else "🔓 Safe Mode Deactivated"
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
    
    private fun showQRCode(qrData: String) {
        // Simplified QR display - in full implementation would show actual QR code
        binding.qrCodeDisplay.text = "QR Code Data:\n$qrData"
        Toast.makeText(this, "QR Code generated - scan with Pi verifier", Toast.LENGTH_LONG).show()
    }
    
    private fun formatDuration(milliseconds: Long): String {
        val seconds = milliseconds / 1000
        val minutes = seconds / 60
        val hours = minutes / 60
        
        return when {
            hours > 0 -> "${hours}h ${minutes % 60}m"
            minutes > 0 -> "${minutes}m ${seconds % 60}s"
            else -> "${seconds}s"
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Camera permission granted", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this, "Camera permission required for QR scanning", Toast.LENGTH_LONG).show()
                }
            }
        }
    }
}