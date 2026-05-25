using System.Diagnostics;
using System.IO;

namespace DafnyCore;

public static class MyStopwatch {
  
  private static readonly Stopwatch Stopwatch = new Stopwatch();
  private static double pluginTime;
  private static double parsingTime;
  private static double resolutionTime;
  private static double verificationTime;
  
  public static void Start() {
    Stopwatch.Start();
  }
  
  public static void Stop() {
    Stopwatch.Stop();
  }

  public static void IncreasePluginTime() {
    pluginTime += Stopwatch.Elapsed.TotalMilliseconds;  
    Stopwatch.Reset();
  }

  public static void SetParsingTime() {
    parsingTime = Stopwatch.Elapsed.TotalMilliseconds;
    Stopwatch.Reset();
  }
  
  public static void IncreaseResolutionTime() {
    resolutionTime += Stopwatch.Elapsed.TotalMilliseconds;  
    Stopwatch.Reset();
  }
  
  public static void SetVerificationTime() {
    verificationTime = Stopwatch.Elapsed.TotalMilliseconds;
    Stopwatch.Reset();
  }

  public static void SaveTimestamps() {
    using var sw = File.AppendText("elapsed-time.csv");
    sw.WriteLine("parsing_time,plugin_time,resolution_time,verification_time,dafny_time");
    var dafnyTime = parsingTime + pluginTime + resolutionTime + verificationTime;
    sw.WriteLine(parsingTime + "," + pluginTime + "," + resolutionTime + "," + verificationTime + "," + dafnyTime);
  }
}