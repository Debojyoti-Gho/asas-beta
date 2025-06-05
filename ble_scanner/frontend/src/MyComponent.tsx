import React from "react"
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib"

const BLEScanner = () => {
  const scanBLE = async () => {
    try {
      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: ['battery_service']
      });

      const result = {
        name: device.name || "Unnamed device",
        id: device.id
      };

      Streamlit.setComponentValue(result);
    } catch (error) {
      Streamlit.setComponentValue({ error: error.message });
    }
  };

  return (
    <div>
      <button onClick={scanBLE}>Scan BLE Devices</button>
    </div>
  );
};

export default withStreamlitConnection(BLEScanner);
