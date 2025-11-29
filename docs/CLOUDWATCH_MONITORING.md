# CloudWatch Monitoring Setup

## Overview
Your CV Location Classifier now has comprehensive CloudWatch monitoring configured through SAM/CloudFormation. This includes logs, metrics, alarms, and a dashboard for complete observability.

## Features Enabled

### 1. **Lambda Insights** 
- Enhanced monitoring layer added to your Lambda function
- Provides detailed metrics on memory usage, CPU time, network, and disk I/O
- **Free Tier:** 1 million requests per month

### 2. **CloudWatch Logs**
- Retention: 7 days (configurable in `template.yaml`)
- Log Group: `/aws/lambda/cv-location-classifier`
- **Free Tier:** 5GB ingestion, 5GB archive per month

### 3. **CloudWatch Alarms**
Six alarms monitoring critical metrics:

| Alarm | Threshold | Period | Description |
|-------|-----------|--------|-------------|
| `cv-location-classifier-errors` | 5 errors | 5 min | Lambda function errors |
| `cv-location-classifier-throttles` | 10 throttles | 5 min | Lambda throttling events |
| `cv-location-classifier-duration` | 50 seconds | 5 min | High execution time |
| `cv-location-classifier-4xx-errors` | 50 errors | 5 min | API Gateway client errors |
| `cv-location-classifier-5xx-errors` | 10 errors | 5 min | API Gateway server errors |

**Free Tier:** 10 alarms per month

### 4. **CloudWatch Dashboard**
Pre-configured dashboard with 5 widgets:
- Lambda invocations, errors, and throttles
- Lambda duration (avg/max)
- API Gateway request counts
- API Gateway latency (avg/p99)
- Recent error logs

## Deployment

Deploy the updated stack with CloudWatch monitoring:

```powershell
# Build and deploy
cd "d:\VS Code\cv_pipeline"
sam build
sam deploy --no-confirm-changeset
```

## Accessing CloudWatch

### 1. **Dashboard**
After deployment, check the CloudFormation outputs:
```powershell
aws cloudformation describe-stacks --stack-name cv-location-classifier --query "Stacks[0].Outputs[?OutputKey=='CloudWatchDashboard'].OutputValue" --output text
```

Or visit: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=cv-location-classifier-monitoring

### 2. **Logs**
View Lambda logs:
```powershell
aws logs tail /aws/lambda/cv-location-classifier --follow
```

Or in the console:
```powershell
aws cloudformation describe-stacks --stack-name cv-location-classifier --query "Stacks[0].Outputs[?OutputKey=='CloudWatchLogs'].OutputValue" --output text
```

### 3. **Alarms**
List all alarms:
```powershell
aws cloudwatch describe-alarms --alarm-name-prefix cv-location-classifier
```

View alarm status:
```powershell
aws cloudwatch describe-alarms --alarm-names cv-location-classifier-errors cv-location-classifier-5xx-errors
```

### 4. **Lambda Insights**
View in the console:
1. Go to Lambda → Functions → cv-location-classifier
2. Click "Monitor" tab
3. Select "View Lambda Insights"

## Querying Logs

### Find Errors
```powershell
aws logs filter-log-events --log-group-name /aws/lambda/cv-location-classifier --filter-pattern "ERROR"
```

### Check Specific Endpoint
```powershell
aws logs filter-log-events --log-group-name /aws/lambda/cv-location-classifier --filter-pattern "/predict"
```

### CloudWatch Insights Query
Run in the AWS Console → CloudWatch → Logs Insights:

**Top 10 Slowest Requests:**
```
fields @timestamp, @duration, @message
| filter @message like /predict/
| sort @duration desc
| limit 10
```

**Error Rate by Hour:**
```
fields @timestamp, @message
| filter @message like /ERROR/
| stats count() as error_count by bin(1h)
```

**Memory Usage:**
```
fields @timestamp, @maxMemoryUsed / 1000000 as memory_mb
| sort @timestamp desc
| limit 50
```

## Setting Up Email Alerts

To receive email notifications when alarms trigger:

### 1. Create SNS Topic
```powershell
aws sns create-topic --name cv-location-classifier-alerts
```

### 2. Subscribe Your Email
```powershell
aws sns subscribe --topic-arn arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:cv-location-classifier-alerts --protocol email --notification-endpoint your-email@example.com
```

### 3. Confirm Subscription
Check your email and click the confirmation link.

### 4. Update Alarms
Add to each alarm in `template.yaml`:
```yaml
AlarmActions:
  - !Ref AlertTopic
```

And add the SNS topic resource:
```yaml
AlertTopic:
  Type: AWS::SNS::Topic
  Properties:
    TopicName: cv-location-classifier-alerts
    DisplayName: CV Location Classifier Alerts
    Subscription:
      - Endpoint: your-email@example.com
        Protocol: email
```

## Monitoring Best Practices

### 1. **Regular Review**
- Check dashboard weekly for trends
- Review alarms monthly to adjust thresholds
- Analyze logs after each deployment

### 2. **Cost Optimization**
- Keep log retention at 7 days (adjustable in template)
- Use CloudWatch Logs Insights instead of exporting logs
- Monitor free tier usage: https://console.aws.amazon.com/billing/home#/freetier

### 3. **Performance Tuning**
- If Duration alarm triggers frequently, optimize Lambda code or increase memory
- If Throttles occur, check concurrent execution limits
- Monitor API Gateway latency to identify bottlenecks

### 4. **Security**
- Review 4xx errors to detect potential attacks
- Monitor unusual traffic patterns in the dashboard
- Set up AWS Config for compliance tracking (optional)

## Key Metrics to Watch

| Metric | What it Means | Action if High |
|--------|---------------|----------------|
| Invocations | Total API calls | Normal growth expected |
| Errors | Function failures | Check logs, fix bugs |
| Throttles | Hitting concurrency limits | Request limit increase |
| Duration | Response time | Optimize code or increase memory |
| 4XX Errors | Client errors (bad requests) | Check API validation |
| 5XX Errors | Server errors | Critical: check logs immediately |

## Free Tier Limits

| Service | Free Tier | Your Usage |
|---------|-----------|------------|
| CloudWatch Logs | 5GB ingestion/month | Low (API logs only) |
| CloudWatch Alarms | 10 alarms/month | 6 alarms |
| Lambda Insights | 1M requests/month | Depends on traffic |
| CloudWatch Dashboards | 3 dashboards/month | 1 dashboard |
| CloudWatch Metrics | 10 custom metrics/month | 0 (using AWS metrics only) |

**Estimated cost beyond free tier:** ~$0-5/month for typical usage

## Troubleshooting

### Dashboard Not Showing Data
- Verify Lambda function is being invoked
- Check alarm status: `aws cloudwatch describe-alarms`
- Wait 5-10 minutes for metrics to populate

### Alarms Not Triggering
- Test by intentionally causing an error
- Verify alarm state: `aws cloudwatch describe-alarm-history --alarm-name cv-location-classifier-errors`
- Check SNS subscription is confirmed (if using email alerts)

### High CloudWatch Costs
- Reduce log retention in `template.yaml`
- Minimize custom metrics
- Use log filtering to reduce storage

## Next Steps

1. **Deploy the updated stack** with CloudWatch monitoring
2. **Test the API** and verify logs are appearing
3. **Set up SNS email alerts** (optional but recommended)
4. **Review the dashboard** after 24 hours of traffic
5. **Adjust alarm thresholds** based on your baseline metrics

## Additional Resources

- [CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
- [Lambda Insights](https://docs.aws.amazon.com/lambda/latest/dg/monitoring-insights.html)
- [CloudWatch Pricing](https://aws.amazon.com/cloudwatch/pricing/)
- [Best Practices for CloudWatch](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Best_Practice_Recommended_Alarms_AWS_Services.html)
