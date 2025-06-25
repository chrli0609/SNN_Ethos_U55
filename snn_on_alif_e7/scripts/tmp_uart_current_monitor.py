#!/usr/bin/env python3
"""
UART Monitor with Current Recording
Monitors UART for trigger message, then records multimeter current for 10 seconds
"""
import msvcrt
import serial
import time
import csv
import threading
import argparse
from datetime import datetime
import statistics
import sys

import re


class UARTCurrentMonitor:
    def __init__(self, uart_port, multimeter_port, uart_baud=115200, mm_baud=115200):
        self.uart_port = uart_port
        self.multimeter_port = multimeter_port
        self.uart_baud = uart_baud
        self.mm_baud = mm_baud
        
        self.uart_conn = None
        self.mm_conn = None
        self.recording = False
        self.stop_monitoring = False
        self.current_readings = []
        
        # Inference timing data collection
        self.collecting_inference_data = False
        self.inference_exe_time = []
        self.avg_current = []
        self.model_names = []

        self.collecting_layer_update_data = False
        self.layer_update_ratio_data = []
        self.tmp_layer_update_data = []


        self.out_filename = None
        
    def connect_uart(self):
        """Connect to UART port"""
        try:
            self.uart_conn = serial.Serial(
                port=self.uart_port,
                baudrate=self.uart_baud,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=0.5,
                #encoding='utf-8',
                #newline='\n'
            )
            print(f"Connected to UART on {self.uart_port}")
            return True
        except Exception as e:
            print(f"Failed to connect to UART: {e}")
            return False
    
    def connect_multimeter(self):
        """Connect to multimeter"""
        try:
            self.mm_conn = serial.Serial(
                port=self.multimeter_port,
                baudrate=self.mm_baud,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=0.1
            )
            print(f"Connected to multimeter on {self.multimeter_port}")
            
            # Test connection with ID query
            response = self.send_scpi_command('*IDN?')
            if response:
                print(f"Multimeter ID: {response}")
                return True
            else:
                print("Multimeter not responding")
                return False
        except Exception as e:
            print(f"Failed to connect to multimeter: {e}")
            return False
    
    def send_scpi_command(self, command, get_response=True):
        """Send SCPI command to multimeter and get response"""
        if not self.mm_conn:
            return None
            
        try:
            cmd = command + '\n'
            self.mm_conn.write(cmd.encode('ascii'))
            
            if get_response:
                # Read response terminated by CR LF
                response = b''
                retries = 0
                while retries < 5:
                    data = self.mm_conn.read(64)
                    if data:
                        response += data
                        if response.endswith(b'\r\n'):
                            break
                    else:
                        retries += 1
                        time.sleep(0.1)
                
                if response:
                    result = response.decode(errors="backslashreplace").strip()
                    return result
                else:
                    return None
            return True
        except Exception as e:
            print(f"SCPI command error: {e}")
            return None
    
    def get_current_measurement(self):
        """Get current measurement from multimeter"""
        try:
            # Get measurement
            meas_str = self.send_scpi_command('MEAS1?')
            if meas_str:
                current = float(meas_str)
                return current
        except ValueError:
            pass
        return None
    
    def monitor_uart(self):
        """Monitor UART for trigger message"""
        print("Monitoring UART for trigger message...")
        print("Press 'q' and Enter to quit")
        
        while not self.stop_monitoring:
            try:
                if self.uart_conn and self.uart_conn.in_waiting > 0:
                    line = self.uart_conn.readline().decode('utf-8', errors='ignore').strip()
                    print(f"UART: {line}")
                    
                    
                    if "model name:" in line:
                        print("ABout to do model name stuff")
                        self.model_names.append(re.sub(r"\s+", "", re.search(r"model name:\s*(.*)", line).group(1)))

                    
                    #Check for inference timing data collection start
                    if "Start printing layer update ratio" in line:
                        print("Starting layer update ratio data collection...")
                        self.start_layer_update_ratio_data_collection()
                    
                    # Check for inference timing data collection stop
                    elif "Stop printing layer update ratio" in line:
                        print("Stopping layer update ratio data collection...")
                        self.stop_layer_update_ratio_data_collection()
                    
                    # Collect inference timing data if we're in collection mode
                    elif self.collecting_layer_update_data:
                        print("line:", line, "\ncapturing:", re.search(r"\d+\.\d+", line))
                        m = re.search(r"\d+\.\d+", line)
                        if m:
                            self.tmp_layer_update_data.append(m.group(0))



                    #if "avg_inference_time_per_sample" in line:
                    #if "avg_inference_exe_time_per_sample" in line:
                    if "avg_inference_time_per_forward_pass" in line:
                        print("About to do avg_inference time stuff", re.search(r"\d+\.\d+", line))
                        self.inference_exe_time.append(re.search(r"\d+\.\d+", line).group(0))
                    
                    
                    # Check for trigger message
                    if "Start running inference forever" in line:
                        print("Trigger detected! Starting measurement sequence...")
                        self.start_measurement_sequence()
                    
                    # Check for end message
                    if line == "End of main() reached":
                        print("End message detected, stopping monitoring")
                        break
                
                ## Check for user input (non-blocking)
                #if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    #user_input = input().strip().lower()
                    #if user_input == 'q':
                        #break
                if msvcrt.kbhit():
                    c = msvcrt.getche().decode()
                    if c.lower() == 'q':
                        break
                        
            except serial.SerialTimeoutException:
                # Timeout is normal, continue monitoring
                pass
            except Exception as e:
                print(f"UART monitoring error: {e}")
                break
            
            time.sleep(0.001)  # Small delay to prevent CPU spinning
    
    def start_measurement_sequence(self):
        """Start the measurement sequence: wait 2s, then record for 10s"""
        if self.recording:
            print("Already recording, ignoring trigger")
            return
        
        print("Waiting 2 seconds before starting recording...")
        time.sleep(2)
        
        print("Starting 10-second current recording...")
        self.recording = True
        self.current_readings = []
        
        # Record for 10 seconds
        start_time = time.time()
        end_time = start_time + 10.0
        
        while time.time() < end_time and not self.stop_monitoring:
            current = self.get_current_measurement()
            if current is not None:
                timestamp = time.time() - start_time
                self.current_readings.append({
                    'timestamp': timestamp,
                    'current': current
                })
                print(f"Current: {current:.6f} A (t={timestamp:.3f}s)")
            
            time.sleep(0.1)  # Sample every 100ms
        
        self.recording = False
        self.save_and_analyze_data()
    
    def save_and_analyze_data(self):
        """Save recorded data to CSV and compute statistics"""
        if not self.current_readings:
            print("No data recorded")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"current_recording_{timestamp}.csv"
        
        
        # Save to CSV
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'current']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for reading in self.current_readings:
                    writer.writerow(reading)
            
            print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return
        
        # Compute statistics
        currents = [reading['current'] for reading in self.current_readings]
        
        avg_current = statistics.mean(currents)
        min_current = min(currents)
        max_current = max(currents)
        std_current = statistics.stdev(currents) if len(currents) > 1 else 0


        self.avg_current.append(avg_current)
        
        print("\n=== Current Measurement Statistics ===")
        print(f"Recording duration: {self.current_readings[-1]['timestamp']:.3f} seconds")
        print(f"Number of samples: {len(currents)}")
        print(f"Average current: {avg_current:.6f} A")
        print(f"Minimum current: {min_current:.6f} A")
        print(f"Maximum current: {max_current:.6f} A")
        print(f"Standard deviation: {std_current:.6f} A")
        print("======================================\n")


    def start_layer_update_ratio_data_collection(self):
        self.collecting_layer_update_data = True
        self.tmp_layer_update_data = []

    def stop_layer_update_ratio_data_collection(self):
        """Stop collecting layer update data and save to CSV"""
        self.collecting_layer_update_data = False
        self.layer_update_ratio_data.append(self.tmp_layer_update_data)
    
        
    
    def start_inference_data_collection(self):
        """Start collecting inference timing data"""
        self.collecting_inference_data = True
        self.inference_data = []
        print("Started collecting inference execution time data...")
    
    def stop_inference_data_collection(self):
        """Stop collecting inference timing data and save to CSV"""
        if not self.collecting_inference_data:
            return
            
        self.collecting_inference_data = False
        
        if self.inference_data:
            self.save_inference_data()
        else:
            print("No inference timing data collected")
    
    def process_inference_line(self, line):
        """Process a line that might contain inference timing data"""
        # Look for lines with format: function_name, time_value,
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    function_name = parts[0].strip()
                    time_value = float(parts[1].strip())
                    
                    # Skip empty function names
                    if function_name:
                        self.inference_data.append({
                            'function_name': function_name,
                            'execution_time': time_value
                        })
                        print(f"Collected: {function_name} -> {time_value}")
                except ValueError:
                    # Not a valid timing line, ignore
                    pass
    
    def save_inference_data(self):
        """Save inference timing data to CSV"""
        if not self.inference_data:
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"inference_timing_{timestamp}.csv"
        
        try:
            with open(filename, 'a', newline='') as csvfile:
                fieldnames = ['function_name', 'execution_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for data in self.inference_data:
                    writer.writerow(data)
            
            print(f"Inference timing data saved to {filename}")
            
            # Compute and display statistics
            times = [data['execution_time'] for data in self.inference_data]
            total_time = sum(times)
            avg_time = statistics.mean(times)
            max_time = max(times)
            min_time = min(times)
            
            #print(f"\n=== Inference Timing Statistics ===")
            #print(f"Number of functions: {len(self.inference_data)}")
            #print(f"Total execution time: {total_time:.6f} ms")
            #print(f"Average function time: {avg_time:.6f} ms")
            #print(f"Max function time: {max_time:.6f} ms")
            #print(f"Min function time: {min_time:.6f} ms")
            
            # Show top 5 slowest functions
            sorted_data = sorted(self.inference_data, key=lambda x: x['execution_time'], reverse=True)
            print(f"\nTop 5 slowest functions:")
            for i, data in enumerate(sorted_data[:5]):
                print(f"  {i+1}. {data['function_name']}: {data['execution_time']:.6f} ms")
            print("===================================\n")
            
        except Exception as e:
            print(f"Error saving inference timing CSV: {e}")
    
    def run(self):
        """Main run loop"""
        # Connect to both ports
        if not self.connect_uart():
            return False
        
        if not self.connect_multimeter():
            return False
        
        try:
            # Start UART monitoring
            self.monitor_uart()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up connections"""
        self.stop_monitoring = True
        
        if self.uart_conn:
            self.uart_conn.close()
            print("UART connection closed")
        
        if self.mm_conn:
            self.mm_conn.close()
            print("Multimeter connection closed")



    def write_results_to_csv(self):

        with open(self.out_filename, 'w') as f:
            writer = csv.writer(f)


            # Write column headers
            writer.writerow(['model', 'inference exe time (us)', 'current usage (A)'])

            for a, b, c, d in zip(self.model_names, self.inference_exe_time, self.avg_current, self.layer_update_ratio_data):
                row = [a, b, c] + d
                writer.writerow(row)

        
# For Windows compatibility (select module)
try:
    import select
except ImportError:
    # Windows doesn't have select for stdin, use alternative approach
    import msvcrt
    
    def check_keyboard_input():
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            return key == 'q'
        return False
    


def main():
    parser = argparse.ArgumentParser(description='UART Monitor with Current Recording')
    parser.add_argument('--uart-port', required=True, 
                       help='UART port (e.g., COM31 on Windows, /dev/ttyUSB0 on Linux)')
    parser.add_argument('--multimeter-port', required=True,
                       help='Multimeter port (e.g., COM7 on Windows, /dev/ttyUSB1 on Linux)')
    parser.add_argument('--uart-baud', type=int, default=115200,
                       help='UART baud rate (default: 115200)')
    parser.add_argument('--mm-baud', type=int, default=115200,
                       help='Multimeter baud rate (default: 115200)')
    
    args = parser.parse_args()
    
    monitor = UARTCurrentMonitor(
        uart_port=args.uart_port,
        multimeter_port=args.multimeter_port,
        uart_baud=args.uart_baud,
        mm_baud=args.mm_baud
    )


    success = monitor.run()
    # Print what I found!
    monitor.out_filename = "inference_and_current.csv"
    monitor.write_results_to_csv()
    if not success:
        monitor.write_results_to_csv()
        sys.exit(1)

if __name__ == "__main__":
    main()