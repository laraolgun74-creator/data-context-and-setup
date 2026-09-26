import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    """
    DataFrames containing all orders as index,
    and various properties of these orders as columns.
    """

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns:
        order_id, wait_time, expected_wait_time,
        delay_vs_expected, order_status
        """

        orders = self.data['orders'].copy()

        if is_delivered:
            orders = orders.query("order_status == 'delivered'").copy()

        orders['order_purchase_timestamp'] = pd.to_datetime(
            orders['order_purchase_timestamp']
        )
        orders['order_estimated_delivery_date'] = pd.to_datetime(
            orders['order_estimated_delivery_date']
        )
        orders['order_delivered_customer_date'] = pd.to_datetime(
            orders['order_delivered_customer_date']
        )

        orders['wait_time'] = (
            orders['order_delivered_customer_date']
            - orders['order_purchase_timestamp']
        ) / np.timedelta64(1, 'D')

        orders['expected_wait_time'] = (
            orders['order_estimated_delivery_date']
            - orders['order_purchase_timestamp']
        ) / np.timedelta64(1, 'D')

        orders['delay_vs_expected'] = (
            orders['wait_time'] - orders['expected_wait_time']
        )

        return orders[
            [
                'order_id',
                'wait_time',
                'expected_wait_time',
                'delay_vs_expected',
                'order_status'
            ]
        ]

    def get_review_score(self):
        """
        Returns:
        order_id, dim_is_five_star, dim_is_one_star,
        review_score, review_cost
        """

        reviews = self.data['order_reviews'][[
            'order_id',
            'review_score'
        ]].copy()

        reviews['dim_is_five_star'] = (
            reviews['review_score'] == 5
        ).astype(int)

        reviews['dim_is_one_star'] = (
            reviews['review_score'] == 1
        ).astype(int)

        reviews['review_cost'] = reviews['review_score'].map({
            1: 100,
            2: 50,
            3: 40,
            4: 0,
            5: 0
        })

        return reviews[
            [
                'order_id',
                'dim_is_five_star',
                'dim_is_one_star',
                'review_score',
                'review_cost'
            ]
        ]

    def get_number_items(self):
        """
        Returns:
        order_id, number_of_items
        """

        return (
            self.data['order_items']
            .groupby('order_id')
            .size()
            .reset_index(name='number_of_items')
        )

    def get_number_sellers(self):
        """
        Returns:
        order_id, number_of_sellers
        """

        return (
            self.data['order_items']
            .groupby('order_id')['seller_id']
            .nunique()
            .reset_index(name='number_of_sellers')
        )

    def get_price_and_freight(self):
        """
        Returns:
        order_id, price, freight_value
        """

        return (
            self.data['order_items']
            .groupby('order_id')
            .agg(
                price=('price', 'sum'),
                freight_value=('freight_value', 'sum')
            )
            .reset_index()
        )

    def get_distance_seller_customer(self):
        """
        Returns:
        order_id, distance_seller_customer
        """

        orders = self.data['orders'][[
            'order_id',
            'customer_id'
        ]]

        customers = self.data['customers'][[
            'customer_id',
            'customer_zip_code_prefix'
        ]]

        sellers = self.data['sellers'][[
            'seller_id',
            'seller_zip_code_prefix'
        ]]

        order_items = self.data['order_items'][[
            'order_id',
            'seller_id'
        ]].drop_duplicates()

        geo = self.data['geolocation']

        customer_geo = (
            customers
            .merge(
                geo,
                left_on='customer_zip_code_prefix',
                right_on='geolocation_zip_code_prefix'
            )
            .groupby('customer_zip_code_prefix')
            [['geolocation_lat', 'geolocation_lng']]
            .mean()
            .reset_index()
        )

        seller_geo = (
            sellers
            .merge(
                geo,
                left_on='seller_zip_code_prefix',
                right_on='geolocation_zip_code_prefix'
            )
            .groupby('seller_zip_code_prefix')
            [['geolocation_lat', 'geolocation_lng']]
            .mean()
            .reset_index()
        )

        customer_geo = customer_geo.rename(columns={
            'geolocation_lat': 'customer_lat',
            'geolocation_lng': 'customer_lng'
        })

        seller_geo = seller_geo.rename(columns={
            'geolocation_lat': 'seller_lat',
            'geolocation_lng': 'seller_lng'
        })

        distances = (
            orders
            .merge(order_items, on='order_id')
            .merge(customers, on='customer_id')
            .merge(customer_geo, on='customer_zip_code_prefix')
            .merge(sellers, on='seller_id')
            .merge(seller_geo, on='seller_zip_code_prefix')
        )

        distances['distance_seller_customer'] = distances.apply(
            lambda row: haversine_distance(
                row['seller_lat'],
                row['seller_lng'],
                row['customer_lat'],
                row['customer_lng']
            ),
            axis=1
        )

        return distances.groupby(
            'order_id',
            as_index=False
        )['distance_seller_customer'].mean()

    def get_training_data(
        self,
        is_delivered=True,
        with_distance_seller_customer=False
    ):
        """
        Returns a clean DataFrame containing order-level features.
        """

        training_set = self.get_wait_time(
            is_delivered=is_delivered
        )

        training_set = training_set.merge(
            self.get_review_score(),
            on='order_id'
        )

        training_set = training_set.merge(
            self.get_number_items(),
            on='order_id'
        )

        training_set = training_set.merge(
            self.get_number_sellers(),
            on='order_id'
        )

        training_set = training_set.merge(
            self.get_price_and_freight(),
            on='order_id'
        )

        if with_distance_seller_customer:
            training_set = training_set.merge(
                self.get_distance_seller_customer(),
                on='order_id'
            )

        training_set = training_set.dropna()

        return training_set
